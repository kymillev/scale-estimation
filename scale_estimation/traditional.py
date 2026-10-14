from __future__ import annotations

import json

import cv2
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from .utils import show_BGR

HOT_PINK_BGR = (180, 105, 255)  # #FF69B4

LINE_SEG_PAD = 20
RULER_POLY_DILATE_PX = 8
RULER_POLY_MIN_OVERLAP = 0.5


def load_ruler_polygon_mask(
    json_path: str,
    shape_hw: tuple[int, int],
    dilate_px: int = RULER_POLY_DILATE_PX,
) -> np.ndarray:
    """Binary mask of the ruler polygon from ``normalized_points`` (optionally dilated)."""
    with open(json_path, "r") as f:
        ruler_data = json.load(f)

    pts = np.asarray(ruler_data["normalized_points"], dtype=np.float32)
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel)

    return mask

def _find_best_line_component(
    bw: np.ndarray,
    gray: np.ndarray,
    pad: int,
    original_shape: tuple[int, int],
    ruler_poly_mask: np.ndarray | None = None,
    visualize: bool = False,
):
    """Find the top-scoring elongated rectangular components."""

    img_h, img_w = gray.shape[:2]
    img_h -= pad * 2
    img_w -= pad * 2

    edge_mask = np.zeros_like(bw, dtype=np.uint8)
    edge_ratio = 0.03
    edge_width = int(edge_ratio * img_w)
    edge_height = int(edge_ratio * img_h)
    edge_mask[: pad + edge_height, :] = 1
    edge_mask[-pad - edge_height :, :] = 1
    edge_mask[:, : pad + edge_width] = 1
    edge_mask[:, -pad - edge_width :] = 1

    if visualize:
        plt.imshow(edge_mask, cmap="gray")
        plt.title("edge_mask")
        plt.show()

    # Restrict connected components to the (padded) ruler polygon when available
    if ruler_poly_mask is not None:
        poly_pad = cv2.copyMakeBorder(
            ruler_poly_mask,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        bw = cv2.bitwise_and(bw, poly_pad)
    else:
        poly_pad = None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    if num <= 1:
        logger.debug("No connected components in binary mask")
        return None, None, []

    scored_components = []
    if visualize:
        plt.figure()
        plt.imshow(bw, cmap="gray")
        plt.title("bw")
        plt.show()
    for i in range(1, num):
        x, y, w, h, area = stats[i]

        # Ignore tiny components
        if area < 0.01 * img_h * img_w:
            continue

        # Ignore very large components
        if area > 0.4 * img_h * img_w:
            continue

        # Ignore components that stretch too much vertically
        if h >= 0.85 * img_h:
            continue

        if h < 0.05 * img_h:
            continue
        if w < 0.3 * img_w:
            continue

        bbox_area = max(1, w * h)

        # How much of the bounding box is actually occupied
        rectangularity = float(area) / float(bbox_area)
        logger.debug(
            "Before clean: area={}, bbox_area={}, rectangularity={:.3f}",
            area,
            bbox_area,
            rectangularity,
        )

        # Long/thin components are preferred
        width_ratio = min(0.75, w / img_w)
        height_ratio = min(0.75, h / img_h)
        elongation = 1 - max(0.1, h / w)
        length_px = float(max(w, h))

        # Prefer symmetrical components
        mid_x = int(round(x + w / 2))
        mask = labels == i
        area_left = np.sum(mask[:, mid_x:])
        area_right = np.sum(mask[:, :mid_x])
        if area_left > area_right:
            area_left, area_right = area_right, area_left

        symmetry_ratio = area_left / area_right

        # Ruler should not be too close to the edge of the image
        edge_overlap = (
            cv2.countNonZero(cv2.bitwise_and(mask.astype(np.uint8), edge_mask)) / area
        )
        # Score
        score = (
            width_ratio
            + elongation
            + (rectangularity**2)
            - height_ratio
            - edge_overlap
            + symmetry_ratio
        )

        component = {
            "component_id": i,
            "score": score,
            "length_px": length_px,
            "rectangularity": rectangularity,
            "elongation": elongation,
            "symmetry_ratio": symmetry_ratio,
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "edge_overlap": edge_overlap,
            "bbox": (
                int(x - pad),
                int(y - pad),
                int(w),
                int(h),
            ),
            "area": int(area),
            "labels": labels,
        }

        scored_components.append(component)

    if not scored_components:
        logger.debug("No line-like connected components after filtering")
        return None, None, []

    # Sort highest score first
    scored_components.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Keep top 3 for debugging
    top_components = scored_components[:3]
    cleaned_components = []

    # Clean top 3 components and rescore
    for component in top_components:
        component_id = component["component_id"]
        mask = component["labels"] == component_id
        clean_mask = clean_angled_line(mask.astype(np.uint8) * 255)
        component["final_mask"] = (
            clean_mask[
                pad : pad + original_shape[0],
                pad : pad + original_shape[1],
            ]
            > 0
        )

        num, labels_, stats, _ = cv2.connectedComponentsWithStats(
            clean_mask.astype(np.uint8), connectivity=8
        )

        if num != 2:
            logger.debug(
                "Found {} components after cleaning component {}, expected 2",
                num,
                component_id,
            )
            continue

        # We should only have 2, background and the line
        # If not then something went wrong or the mask is not the ruler line
        x, y, w, h, area = stats[1]
        mask = labels_ == 1
        component["final_mask"] = mask[
            pad : pad + original_shape[0],
            pad : pad + original_shape[1],
        ]

        bbox_area = max(1, w * h)

        # How much of the bounding box is actually occupied
        rectangularity = float(area) / float(bbox_area)
        logger.debug(
            "After clean: area={}, bbox_area={}, rectangularity={:.3f}",
            area,
            bbox_area,
            rectangularity,
        )

        # Long/thin components are preferred
        width_ratio = min(0.75, w / img_w)
        height_ratio = min(0.75, h / img_h)
        elongation = 1 - max(0.1, h / w)
        length_px = float(max(w, h))

        # Prefer symmetrical components
        mid_x = int(round(x + w / 2))

        area_left = np.sum(mask[:, mid_x:])
        area_right = np.sum(mask[:, :mid_x])
        if area_left > area_right:
            area_left, area_right = area_right, area_left

        symmetry_ratio = area_left / area_right

        # Ruler should not be too close to the edge of the image
        edge_overlap = (
            cv2.countNonZero(cv2.bitwise_and(mask.astype(np.uint8), edge_mask)) / area
        )

        # Score
        score = (
            width_ratio
            + elongation
            + (rectangularity**2)
            - height_ratio
            - edge_overlap
            + symmetry_ratio
        )
        component["score"] = score
        component["elongation"] = elongation
        component["rectangularity"] = rectangularity
        component["width_ratio"] = width_ratio
        component["height_ratio"] = height_ratio
        component["symmetry_ratio"] = symmetry_ratio
        component["edge_overlap"] = edge_overlap
        component["area"] = area
        cleaned_components.append(component)

    if not cleaned_components:
        logger.debug("No components remained after line cleaning")
        return None, None, []

    # Sort highest score first
    cleaned_components.sort(
        key=lambda x: x["score"],
        reverse=True,
    )
    # Best component
    best = cleaned_components[0]
    best_i = best["component_id"]

    mask_pad = labels == best_i

    mask = mask_pad[
        pad : pad + original_shape[0],
        pad : pad + original_shape[1],
    ]

    return mask, best, cleaned_components


def segment_line_connected_components(
    img_bgr: np.ndarray,
    pad: int = LINE_SEG_PAD,
    ruler_poly_mask: np.ndarray | None = None,
    visualize: bool = False,
):
    """
    Segment the dominant scale-bar line via connected components.

    Works for both:
      - white foreground on black ruler
      - black foreground on white ruler

    Returns:
        (mask_bool, info, debug_components) or (None, None, [])
    """

    img_pad = cv2.copyMakeBorder(
        img_bgr,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Try BOTH polarities
    thresholds = [
        (
            "inv",
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                5,
            ),
        ),
        (
            "normal",
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            ),
        ),
    ]

    candidates = []

    for polarity, bw in thresholds:

        mask, info, top_components = _find_best_line_component(
            bw=bw,
            gray=gray,
            pad=pad,
            original_shape=img_bgr.shape[:2],
            ruler_poly_mask=ruler_poly_mask,
            visualize=visualize,
        )

        if info is not None:
            info["polarity"] = polarity

            for component in top_components:
                component["polarity"] = polarity

            candidates.append((mask, info, top_components))

    if not candidates:
        logger.warning("No line segment candidates found")
        return None, None, []

    # Pick best candidate from either polarity
    mask, best, top_components = max(
        candidates,
        key=lambda x: x[1]["score"],
    )

    # Combine all top candidates from both polarities for debugging
    all_debug_components = []

    for _, _, components in candidates:
        all_debug_components.extend(components)

    all_debug_components.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Keep global top 3
    all_debug_components = all_debug_components[:3]

    logger.debug(
        "Selected line component score={:.2f} polarity={} length_px={:.1f}",
        best["score"],
        best.get("polarity"),
        best["length_px"],
    )

    if visualize:
        vis = overlay_segmented_line(img_bgr, mask)
        show_BGR(vis)
        plt.title(
            f"selected CC | score={best['score']:.0f} | "
            f"length={best['length_px']:.0f}px | polarity={best.get('polarity')}"
        )
        plt.show()

        for rank, component in enumerate(all_debug_components, start=1):
            final_mask = component.get("final_mask")
            if final_mask is None:
                continue
            vis = overlay_segmented_line(img_bgr, final_mask)
            show_BGR(vis)
            plt.title(
                f"#{rank} final mask | score={component['score']:.2f} | "
                f"polarity={component.get('polarity')}"
            )
            plt.show()

    return mask, best, all_debug_components


def overlay_segmented_line(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr=HOT_PINK_BGR,
):
    """Paint segmented line pixels hot pink on a copy of the image."""
    out = img_bgr.copy()
    out[mask] = color_bgr
    return out


def visualize_top_components(
    img_bgr: np.ndarray,
    components: list[dict],
):
    """Visualize the top-scoring connected components."""

    vis = img_bgr.copy()

    for rank, component in enumerate(components, start=1):

        x, y, w, h = component["bbox"]

        # Clip bbox to image
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_bgr.shape[1] - 1, x + w)
        y2 = min(img_bgr.shape[0] - 1, y + h)

        # Different line thickness makes ranking easy to see
        thickness = 3 if rank == 1 else 2

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            HOT_PINK_BGR,
            thickness,
        )

        label = f"#{rank} {component['score']:.0f} {component['polarity']}"

        # Put label above bbox when possible
        text_y = max(15, y1 - 5)

        cv2.putText(
            vis,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HOT_PINK_BGR,
            2,
            cv2.LINE_AA,
        )

    return vis


def clean_angled_line(mask):
    # 1. Find all coordinates of the line pixels
    coords = cv2.findNonZero(mask)

    # 2. Get the oriented bounding box to find the angle
    # rect = ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(coords.astype(np.float32))

    angle = rect[-1]

    # OpenCV's angle logic can be tricky; we want the line to be horizontal.
    # Adjust angle so we rotate the 'long' side to the horizontal axis.
    (h, w) = mask.shape[:2]
    center = (w // 2, h // 2)

    if rect[1][0] < rect[1][1]:  # width < height
        angle = angle - 90

    # 3. Rotate the image to make the line horizontal
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # 4. Apply Horizontal Morphology on the straightened line
    # The kernel width should be larger than the tick height/thickness
    kernel = np.ones((1, 50), np.uint8)
    cleaned_rotated = cv2.morphologyEx(rotated_mask, cv2.MORPH_OPEN, kernel)

    # 5. Rotate back to original orientation
    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    final_mask = cv2.warpAffine(cleaned_rotated, M_inv, (w, h), flags=cv2.INTER_NEAREST)

    return final_mask


def mask_end_midpoints(mask: np.ndarray):
    """
    Left/right endpoints of a line mask.

    Left: midpoint of the top-leftmost and bottom-leftmost pixels.
    Right: midpoint of the top-rightmost and bottom-rightmost pixels.
    """
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        logger.warning("Cannot compute end midpoints: empty line mask")
        return None, None

    min_x = int(xs.min())
    max_x = int(xs.max())

    left_ys = ys[xs == min_x]
    right_ys = ys[xs == max_x]

    left = np.array(
        [min_x, 0.5 * (float(left_ys.min()) + float(left_ys.max()))],
        dtype=np.float32,
    )
    right = np.array(
        [max_x, 0.5 * (float(right_ys.min()) + float(right_ys.max()))],
        dtype=np.float32,
    )
    return left, right
