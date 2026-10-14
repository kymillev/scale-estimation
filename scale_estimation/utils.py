import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

type RGBImage = np.ndarray[tuple[Any, Any, Literal[3]], np.dtype[np.uint8]]

COLOR_PALETTE = [
    (180, 119, 31),  # Blue
    (14, 127, 255),  # Orange
    (44, 160, 44),  # Green
    (40, 39, 214),  # Red
    (189, 103, 148),  # Purple
    (75, 86, 140),  # Brown
    (194, 119, 227),  # Pink
    (127, 127, 127),  # Gray
    (34, 189, 188),  # Olive
    (207, 190, 23),  # Cyan
]


def show_BGR(img: np.ndarray):
    return plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def sorted_alphanumeric(items: Iterable[str]) -> list[Any]:
    def convert(text: str):
        return int(text) if text.isdigit() else text

    def alphanum_key(key: str):
        return [convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(items, key=alphanum_key)


def load_rgb_image(path: str | Path) -> RGBImage | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def get_scaled_size(img_shape, base_size=10, ref_diag=2202.9):
    h, w = img_shape[:2]
    img_diag = np.sqrt(w**2 + h**2)
    if ref_diag == 0:
        return base_size
    scale_factor = img_diag / ref_diag
    scaled_size = base_size * scale_factor
    return max(1, round(scaled_size))


def ensure_horizontal_image(img: RGBImage) -> RGBImage:
    """
    Ensure the image is horizontal by rotating it 90 degrees clockwise if its height is greater than its width.

    Args:
        img (RGBImage): The image to ensure is horizontal
    Returns:
        RGBImage: The image ensured to be horizontal
    """
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# Source: https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
def rotate_image(img: np.ndarray, angle: float):
    """
    Rotate an image by a given angle

    Args:
        img (np.ndarray): The image to rotate
        angle (float): The angle to rotate the image by in degrees

    Returns:
        np.ndarray: The rotated image
        np.ndarray: The transformation matrix
    """
    size_reverse = np.array(img.shape[1::-1])  # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.0), angle, 1.0)
    MM = np.absolute(M[:, :2])
    size_new = MM @ size_reverse
    M[:, -1] += (size_new - size_reverse) / 2.0
    return cv2.warpAffine(img, M, tuple(size_new.astype(int))), M


def visualize_ruler_crop(
    img: RGBImage,
    ruler_data: dict[str, Any] | list[dict[str, Any]],
    kind: Literal["original", "crop", "normalized"] = "normalized",
    tick_size: int | None = None,
    rotated: bool = False,
):
    out = img.copy()

    if tick_size is None:
        tick_size = get_scaled_size(img.shape)

    if kind == "original":
        points_key = "orig_points"
    elif kind == "crop":
        points_key = "crop_points"
    else:
        points_key = "normalized_points"

    if isinstance(ruler_data, dict):
        tick_labels = ruler_data.get("tick_labels", [])
    else:
        tick_labels = ruler_data

    h, w = out.shape[:2]
    if rotated:
        out = cv2.rotate(out, cv2.ROTATE_180)

    # Draw tick points shape (N,2)
    for i, tick_label in enumerate(tick_labels):
        if not isinstance(tick_label, dict):
            tick_label = tick_label.to_dict()
        tick_pts = np.array(tick_label[points_key], dtype=np.int32)
        if rotated:
            tick_pts = np.array([w - 1, h - 1], dtype=np.int32) - tick_pts
        color = COLOR_PALETTE[i]
        for pt in tick_pts:
            cv2.circle(out, pt, tick_size, color, -1)
    return out


SUPPORTED_UNITS = ("mm", "cm", "m", "µm", "inch", "ft")

# Factors that convert one of each unit into millimeters
_UNIT_TO_MM = {
    "m": 1000.0,
    "cm": 10.0,
    "mm": 1.0,
    "µm": 0.001,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "feet": 304.8,
}


def _unit_to_mm_factor(unit: str) -> float:
    try:
        return _UNIT_TO_MM[unit]
    except KeyError:
        supported = ", ".join(SUPPORTED_UNITS)
        raise ValueError(f"Unknown unit: {unit}. Supported units are: {supported}")


def convert_unit(value: float, unit: str, target_unit: str = "mm") -> float:
    """Convert a measurement from one unit to another.

    Values are converted to millimeters, then to ``target_unit``.
    Supported units are ``mm, cm, m, µm, in, ft``

    Args:
        value: The value to convert
        unit: The unit of the value
        target_unit: The target unit, defaults to "mm"

    Returns:
        The converted value
    """
    if target_unit == unit:
        return value

    value_mm = value * _unit_to_mm_factor(unit)
    return value_mm / _unit_to_mm_factor(target_unit)


def calculate_pixel_distance(
    points: np.ndarray, tick_label: str, target_unit: str = "mm"
):
    """
    Calculate the pixel distance between ruler tick mark coordinates and the standard error
    in the target unit.
    Args:
        points (np.ndarray): The points to calculate the pixel distance between
        tick_label (str): The tick label to parse the measurement from
        target_unit (str): The target unit to convert the pixel distance to, defaults to mm

    Returns:
        float: The average pixel distance between succeeding tick marks
        float: The standard deviation of the pixel distance
    """

    # Calculatye succeeding point pixel distances
    differences = np.diff(points, axis=0)

    # To Euclidean distance
    sequential_distances = np.linalg.norm(differences, axis=1)
    avg_dist = float(np.mean(sequential_distances))
    std_dist = float(np.std(sequential_distances))

    # Replace tick label suffixes
    tick_label = (
        tick_label.replace("_partial", "").replace("_curved", "").replace("_inset", "")
    )
    # Parse the tick label (e.g. ticks_5cm, ticks_0.1µm, ticks_1ft)
    pattern = re.compile(r"^ticks_(\d*\.\d+|\d+)(.+)$")
    match = pattern.fullmatch(tick_label.strip())
    if match:
        measure = float(match.group(1))
        unit = match.group(2)
    else:
        raise ValueError(
            f"Unable to parse tick label: {tick_label}. "
            "Ensure it's of the format: ticks_{value}{unit}, where value is a number"
        )

    # Convert the measure to the target unit
    measure = convert_unit(measure, unit, target_unit)

    avg_dist /= measure
    std_dist /= measure

    return avg_dist, std_dist


def convert_coco_to_labelme(coco_json_path, output_path):

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    image_info = coco_data["images"][0]
    file_name = image_info["file_name"]
    width = image_info["width"]
    height = image_info["height"]

    # Map category id → name
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    shapes = []

    for ann in coco_data["annotations"]:
        segmentation = ann["segmentation"][0]
        category_id = ann["category_id"]
        label = categories[category_id]

        # Convert flat list → [[x,y], ...]
        points = [
            [segmentation[i], segmentation[i + 1]]
            for i in range(0, len(segmentation), 2)
        ]

        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }

        shapes.append(shape)

    labelme_output = {
        "version": "5.6.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name,
        "imageHeight": height,
        "imageWidth": width,
        "imageData": None,
    }

    with open(output_path, "w") as f:
        json.dump(labelme_output, f)


def coco_segmentation_to_points(segmentation: list) -> np.ndarray:
    """Convert a COCO polygon segmentation to an (N, 2) array of points."""
    if not segmentation:
        return np.zeros((0, 2), dtype=np.float32)
    poly = (
        segmentation[0] if isinstance(segmentation[0], (list, tuple)) else segmentation
    )
    pts = np.asarray(poly, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    return pts


def load_coco_ruler_annotations(
    coco_data: dict[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Load COCO annotations, largest/highest-score first."""
    if isinstance(coco_data, (str, Path)):
        with open(coco_data, encoding="utf-8") as f:
            coco_data = json.load(f)
    annotations = list(coco_data.get("annotations") or [])
    annotations.sort(
        key=lambda a: float(a.get("score", a.get("area", 0.0))),
        reverse=True,
    )
    return annotations


def annotation_polygon(ann: dict[str, Any]) -> np.ndarray:
    """Polygon points from a COCO annotation (segmentation, else bbox)."""
    segmentation = ann.get("segmentation") or []
    if segmentation and not isinstance(segmentation, dict):
        return coco_segmentation_to_points(segmentation)
    bbox = ann.get("bbox")
    if bbox is None:
        return np.zeros((0, 2), dtype=np.float32)
    x, y, w, h = bbox
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def crop_ruler_from_polygon(
    img: RGBImage,
    polygon: np.ndarray,
    mask_outside: bool = False,
    pad: int = 5,
) -> tuple[RGBImage, tuple[int, int, int, int]]:
    """Crop a ruler from ``img`` using a polygon.

    If ``pad > 0``, the polygon mask is dilated with a symmetric ``pad`` x ``pad``
    kernel (default 5x5) before the bounding box is taken.
    Pixels outside the (dilated) polygon are kept unless ``mask_outside`` is True.

    Returns the bbox crop and the xyxy bbox in original
    image coordinates.
    """
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    img_h, img_w = img.shape[:2]
    x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))

    margin = pad // 2 if pad > 0 else 0
    x0 = int(max(0, x - margin))
    y0 = int(max(0, y - margin))
    x1 = int(min(img_w, x + w + margin))
    y1 = int(min(img_h, y + h + margin))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Ruler polygon bbox is empty or outside the image")

    mask = None
    if len(pts) >= 3:
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        rel = pts.copy()
        rel[:, 0] -= x0
        rel[:, 1] -= y0
        cv2.fillPoly(mask, [np.round(rel).astype(np.int32)], 255)
        if pad > 0:
            kernel = np.ones((pad, pad), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel)
        mx, my, mw, mh = cv2.boundingRect(mask)
        if mw > 0 and mh > 0:
            x0 = x0 + int(mx)
            y0 = y0 + int(my)
            x1 = x0 + int(mw)
            y1 = y0 + int(mh)
            mask = mask[my : my + mh, mx : mx + mw]

    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(img_w, x1))
    y1 = int(min(img_h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Ruler polygon bbox is empty or outside the image")

    crop = img[y0:y1, x0:x1].copy()
    if mask_outside and mask is not None:
        if mask.shape[:2] != crop.shape[:2]:
            mask = mask[: crop.shape[0], : crop.shape[1]]
        crop = cv2.bitwise_and(crop, crop, mask=mask)
    return crop, (x0, y0, x1, y1)


def crop_rulers_from_coco(
    img: RGBImage,
    coco_data: dict[str, Any] | str | Path,
    mask_outside: bool = False,
    pad: int = 5,
) -> list[tuple[RGBImage, tuple[int, int, int, int], dict[str, Any]]]:
    """Crop every ruler polygon in a COCO JSON from ``img``."""
    crops = []
    for ann in load_coco_ruler_annotations(coco_data):
        polygon = annotation_polygon(ann)
        if len(polygon) < 3:
            continue
        crop, bbox = crop_ruler_from_polygon(
            img, polygon, mask_outside=mask_outside, pad=pad
        )
        crops.append((crop, bbox, ann))
    return crops


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest foreground blob in a binary mask."""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    n_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if n_labels <= 2:
        return mask_u8.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def polygons_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Polygon of the largest connected component in a detection mask."""
    from supervision.detection.utils.converters import mask_to_polygons

    component = largest_connected_component(mask)
    polygons = mask_to_polygons(component.astype(bool))
    valid = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            valid.append(pts)
    if not valid:
        return []
    areas = [abs(float(cv2.contourArea(pts))) for pts in valid]
    return [valid[int(np.argmax(areas))]]


def detections_to_coco(
    fname: str,
    width: int,
    height: int,
    detections: Any,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a one-image COCO dict from a supervision ``Detections`` result."""
    if categories is None:
        categories = [{"id": 0, "name": "ruler", "supercategory": "object"}]
    coco = {
        "images": [{"id": 1, "file_name": fname, "width": width, "height": height}],
        "annotations": [],
        "categories": categories,
    }
    if detections is None or len(detections) == 0:
        return coco

    annotations = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = map(float, detections.xyxy[i])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        segmentation = []
        area = float((x2 - x1) * (y2 - y1))
        if detections.mask is not None:
            polygons = polygons_from_mask(detections.mask[i])
            for poly in polygons:
                flat = poly.reshape(-1).astype(float).tolist()
                if len(flat) >= 6:
                    segmentation.append(flat)
            if segmentation:
                pts = polygons[0]
                x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
                bbox = [float(x), float(y), float(w), float(h)]
                area = abs(float(cv2.contourArea(pts)))
        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        score = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 1.0
        )
        annotations.append(
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "score": score,
            }
        )
    coco["annotations"] = annotations
    return coco


def crop_points_to_original(
    points: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Map points from a horizontal ruler crop back to original image coordinates.

    If the unrotated bbox is taller than it is wide, the crop was rotated 90°
    clockwise (``ensure_horizontal_image`` / ``cv2.ROTATE_90_CLOCKWISE``).
    """
    x1, y1, x2, y2 = bbox
    h_crop, w_crop = y2 - y1, x2 - x1
    pts = np.array(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    if h_crop > w_crop:
        x_h, y_h = pts[:, 0], pts[:, 1]
        x_crop = y_h
        y_crop = (h_crop - 1) - x_h
        pts = np.stack([x_crop, y_crop], axis=1)
    return pts + np.array([x1, y1], dtype=float)


RULER_TYPES = {
    "Bausch-Lomb": "Bausch-Lomb",
    "BIG": "BIG",
    "block-100mm": "block-10mm",
    "block-150mm": "block-10mm",
    "block-200mm": "block-10mm",
    "block-1cm": "block-1cm",
    "block-2": "block-2",
    "block-4": "block-4",
    "block-5": "block-5",
    "block-8": "block-8",
    "block-10": "block-10",
    "block-complex_white": "block-complex_white",
    "block-complex": "block-complex",
    "block-vertical": "block-vertical",
    "Calibrite": "Calibrite",
    "card-cm-inch": "card-cm-inch",
    "card-mm-inch": "card-mm-inch",
    "complex-BYU": "complex-BYU",
    "Fisher": "Fisher",
    "GT10cm": "GT",
    "GT5cm": "GT",
    "GT7cm": "GT",
    "Helix": "Helix",
    "jim": "jim",
    "Kodak": "Kodak",
    "MeiseBG": "MeiseBG",
    "NYBG-10": "NYBG-10",
    "QPCard": "QPCard",
    "Sorvall": "Sorvall",
    "strip": "strip",
    "TE263": "TE263",
    "tick-10": "tick-10",
    "tick-5": "tick-5",
    "tick-cm-inch": "tick-cm-inch",
    "tick-cm_half": "tick-cm_half",
    "tick-cm_quarters": "tick-cm_quarters",
    "tick-Danespicta": "tick-Danespicta",
    "tick-Grobet-Vigor": "tick-Grobet-Vigor",
    "tick-Illinois": "tick-Illinois",
    "tick-inch-4": "tick-inch-4",
    "tick-inch-mm": "tick-inch-mm",
    "tick-inch": "tick-inch",
    "tick-mm-inch": "tick-mm-inch",
    "tick-mm-vertical": "tick-mm-vertical",
    "tick-mm": "tick-mm",
    "tick-Sachverstandig": "tick-Sachverstandig",
    "tick-SAFET": "tick-SAFET",
    "tick-Starrett": "tick-Starrett",
    "tick-WhiBal": "tick-WhiBal",
    "tick_Carolina": "tick_Carolina",
    "Tiffen": "Tiffen",
    "Westcott": "Westcott",
    "Xrite-vertical": "Xrite-vertical",
    "Xrite": "Xrite",
    # Generic types
    "block-tick": "block-tick",
    "block": "block",
    "line": "line",
    "line-tick": "line-tick",
    "tick": "tick",
}


def get_ruler_type(name: str):
    """
    Returns the ruler type for the given name

    Args:
        name (str): The name of the ruler

    Returns:
        str: The type of the ruler
    """
    name = name.replace("ruler_", "")
    name = name.replace("-partial-", "_partial_").replace("-partial", "_partial")
    name = name.replace("-curved-", "_curved_").replace("-curved", "_curved")

    for key, value in RULER_TYPES.items():
        if key in name:
            return value

    return name
