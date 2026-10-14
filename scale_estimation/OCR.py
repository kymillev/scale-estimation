from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
from paddleocr import PaddleOCR

from .traditional import (
    load_ruler_polygon_mask,
    mask_end_midpoints,
    segment_line_connected_components,
)
from .utils import calculate_pixel_distance, show_BGR

# Supported length units (and common OCR / spelling variants)
UNIT_ALIASES = {
    "m": "m",
    "dm": "dm",
    "cm": "cm",
    "mm": "mm",
    "µm": "µm",
    "um": "µm",
    "inch": "inch",
    "inches": "inch",
    "in": "inch",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
}

_UNIT_ALT = "|".join(sorted(map(re.escape, UNIT_ALIASES), key=len, reverse=True))
MEASUREMENT_RE = re.compile(
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT_ALT})\b",
    flags=re.IGNORECASE,
)
NUMBER_TOKEN_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
UNIT_TOKEN_RE = re.compile(rf"^({_UNIT_ALT})$", flags=re.IGNORECASE)

_ocr = None


def get_ocr() -> PaddleOCR:
    """Lazy singleton so importing this module does not load PaddleOCR."""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            engine="transformers",
            ocr_version="PP-OCRv6",
        )
    return _ocr


def _load_image(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        path = Path(image_or_path)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img, str(path)
    if isinstance(image_or_path, np.ndarray):
        return image_or_path, None
    raise TypeError("image_or_path must be a path or a numpy array")


def _texts_from_ocr_result(result):
    """Normalize PaddleOCR 3.x predict() output to a list of (text, score)."""
    texts = []
    for res in result:
        data = res.json if hasattr(res, "json") else res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        rec_texts = getattr(res, "rec_texts", None) or (data or {}).get("rec_texts", [])
        rec_scores = getattr(res, "rec_scores", None) or (data or {}).get(
            "rec_scores", []
        )
        for i, text in enumerate(rec_texts):
            score = float(rec_scores[i]) if i < len(rec_scores) else None
            texts.append((str(text), score))
    return texts


def _canonical_unit(raw_unit: str):
    if raw_unit in UNIT_ALIASES:
        return UNIT_ALIASES[raw_unit]
    return UNIT_ALIASES.get(raw_unit.lower())


def parse_length_measurements(text: str):
    """Extract measurements from OCR text like '5cm', '1 cm', '3 µm'."""
    found = []
    for m in MEASUREMENT_RE.finditer(text):
        unit = _canonical_unit(m.group("unit"))
        if unit is None:
            continue
        found.append(
            {
                "value": float(m.group("value").replace(",", ".")),
                "unit": unit,
                "raw": m.group(0),
            }
        )
    return found


def _ndigits(num_str: str) -> int:
    return sum(ch.isdigit() for ch in num_str)


def assemble_measurements_from_ocr_lines(lines):
    """
    Parse value+unit from OCR lines, including fragmented tokens.

    Handles cases like ['10', '0', 'cm'] -> 10 cm (prefer multi-digit
    numbers over stray single digits / tick marks).
    """
    measurements = []
    for text, score in lines:
        for item in parse_length_measurements(text):
            measurements.append(
                {**item, "text": text, "score": score, "source": "ocr_inline"}
            )
    if measurements:
        return measurements

    numbers, units = [], []
    for text, score in lines:
        token = text.strip()
        if not token:
            continue
        if NUMBER_TOKEN_RE.fullmatch(token):
            numbers.append(
                {
                    "value": float(token.replace(",", ".")),
                    "raw": token,
                    "score": score if score is not None else 0.0,
                    "ndigits": _ndigits(token),
                }
            )
        else:
            unit = _canonical_unit(token) if UNIT_TOKEN_RE.fullmatch(token) else None
            if unit is not None:
                units.append(
                    {
                        "unit": unit,
                        "raw": token,
                        "score": score if score is not None else 0.0,
                    }
                )

    if not numbers or not units:
        return []

    # Prefer multi-digit values (e.g. '10' over a stray '0' tick), then score, then magnitude.
    candidates = [n for n in numbers if n["value"] != 0] or numbers
    best_num = max(
        candidates,
        key=lambda n: (n["ndigits"] >= 2, n["ndigits"], n["score"], n["value"]),
    )
    best_unit = max(units, key=lambda u: u["score"])
    raw = f"{best_num['raw']}{best_unit['raw']}"
    score = None
    if best_num["score"] and best_unit["score"]:
        score = min(best_num["score"], best_unit["score"])
    return [
        {
            "value": best_num["value"],
            "unit": best_unit["unit"],
            "raw": raw,
            "text": " ".join(t for t, _ in lines),
            "score": score,
            "source": "ocr_assembled",
        }
    ]


def _enhance_for_ocr(img_bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Pad +CLAHE to recover tiny digits (e.g. '1' in '1cm')."""
    pad = 48
    img_bgr = cv2.copyMakeBorder(
        img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def detect_ruler_lengths(image_or_path, visualize: bool = False):
    """
    OCR length labels on a ruler crop; fall back to enhanced OCR, then CV.

    Returns dict with ocr_lines, measurements, and optional fallback info.
    """
    img, path = _load_image(image_or_path)
    ocr_input = path if path is not None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    title = path or "ndarray"

    ocr = get_ocr()
    lines = _texts_from_ocr_result(ocr.predict(ocr_input))
    measurements = assemble_measurements_from_ocr_lines(lines)
    method = "ocr"
    fallback = None

    if not lines:
        logger.warning("No OCR text found for {}", title)
    else:
        logger.debug("OCR lines for {}: {}", title, lines)

    # Retry on enhanced crop if we didn't get a full value+unit
    if not measurements:
        logger.debug(
            "No length measurement parsed; retrying with enhanced OCR on {}", title
        )
        enhanced = _enhance_for_ocr(img)
        lines = _texts_from_ocr_result(ocr.predict(enhanced))
        measurements = assemble_measurements_from_ocr_lines(lines)
        method = "enhanced_ocr"
        if lines:
            logger.debug("Enhanced OCR lines for {}: {}", title, lines)

    if measurements:
        logger.debug("Parsed measurements for {}: {}", title, measurements)
    else:
        unit_hint = None
        for text, _ in lines:
            if UNIT_TOKEN_RE.fullmatch(text.strip()):
                unit_hint = _canonical_unit(text.strip())
                break
        logger.warning(
            "No length measurement found for {} (unit hint={}); using CV fallback",
            title,
            unit_hint,
        )

        # TODO: Use LLM as fallback
        fallback = None
        if fallback is None:
            logger.warning("CV fallback also found no scale bar for {}", title)
        else:
            logger.debug("CV fallback for {}: {}", title, fallback)

    if visualize:
        show_BGR(img)
        label = measurements or fallback or "none"
        plt.title(f"lengths: {label} | {title}")
        plt.show()

    return {
        "ocr_lines": lines,
        "measurements": measurements,
        "method": method,
        "fallback": fallback,
    }


def _measurement_tick_label(measurement: dict) -> str:
    value = measurement["value"]
    unit = measurement["unit"]
    if float(value).is_integer():
        value_s = str(int(value))
    else:
        value_s = str(value)
    return f"ticks_{value_s}{unit}"


def predict_line_OCR(
    query_img_path,
    visualize: bool = False,
    ruler_poly_mask=None,
):
    """Estimate a line-ruler conversion factor via OCR + traditional line segmentation.

    Returns the same list-of-dicts format as ``predict_tick_marks``, or ``None``
    if OCR / line segmentation does not yield a valid measurement.
    """
    img, path = _load_image(query_img_path)
    title = path or "ndarray"

    ocr_result = detect_ruler_lengths(
        img if path is None else path, visualize=visualize
    )
    measurements = ocr_result.get("measurements") or []
    if not measurements:
        logger.warning("No valid OCR measurement for {}", title)
        return None

    meas = max(
        measurements,
        key=lambda m: m["score"] if m.get("score") is not None else 0.0,
    )
    if meas.get("value") in (None, 0):
        logger.warning(
            "OCR measurement has invalid value {} for {}",
            meas.get("value"),
            title,
        )
        return None

    if ruler_poly_mask is None and path is not None:
        json_path = Path(path).with_suffix(".json")
        if json_path.exists():
            try:
                ruler_poly_mask = load_ruler_polygon_mask(str(json_path), img.shape[:2])
            except Exception as e:
                logger.debug("Could not load ruler polygon for {}: {}", title, e)

    mask, info, _debug = segment_line_connected_components(
        img,
        ruler_poly_mask=ruler_poly_mask,
        visualize=visualize,
    )
    if mask is None or info is None:
        logger.warning("Line segmentation failed for {} after OCR", title)
        return None

    final_mask = info.get("final_mask", mask)
    left_pt, right_pt = mask_end_midpoints(final_mask)
    if left_pt is None or right_pt is None:
        logger.warning("Could not get line end midpoints for {}", title)
        return None

    points = np.stack([left_pt, right_pt], axis=0)
    tick_label = _measurement_tick_label(meas)
    try:
        avg_dist, std_dist = calculate_pixel_distance(points, tick_label=tick_label)
    except ValueError as e:
        logger.warning(
            "Could not compute pixel distance for {} ({}): {}",
            title,
            tick_label,
            e,
        )
        return None

    logger.debug(
        "Line OCR CF for {}: {} -> {:.4f} px/mm (length={:.1f}px)",
        title,
        meas.get("raw"),
        avg_dist,
        float(np.linalg.norm(right_pt - left_pt)),
    )

    gt_tick_label = {
        "label": tick_label,
        "normalized_points": points.tolist(),
    }
    score = meas.get("score")
    return [
        {
            "transformed_points": [points],
            "gt_tick_labels": [gt_tick_label],
            "overall_score": float(score) if score is not None else 1.0,
            "num_matches": 2,
            "used_rotation": False,
            "skip_gt": False,
            "avg_dist": float(avg_dist),
            "std_dist": float(std_dist),
            "source": "line_ocr",
        }
    ]
