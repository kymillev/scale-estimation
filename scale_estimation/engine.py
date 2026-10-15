from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as F
from annoy import AnnoyIndex
from loguru import logger
from PIL import Image
from torchvision import transforms

from .utils import (
    RGBImage,
    calculate_pixel_distance,
    convert_unit,
    crop_points_to_original,
    crop_rulers_from_coco,
    detections_to_coco,
    ensure_horizontal_image,
    get_ruler_type,
    load_rgb_image,
    numpy_image_to_torch,
)

# Minimum and maximum pixel distance in pixels per mm.
# If the prediction exceeds this range, it is considered invalid.
MAX_PIXEL_DIST_MM = 200
MIN_PIXEL_DIST_MM = 0.1

# Maximum std. dev. divided by predicted CF distance for keypoint matches.
# Tick groups may vary at most 10% of the predicted scale factor.
CF_STD_THRESH = 0.1

# Default SI unit for the scale factor
DEFAULT_SI_UNIT = "mm"

# DaD/VGG downsamples 4×; LoMa detector inputs must be divisible by this (and at least this large).
LOMA_SIZE_MULTIPLE = 16

DEFAULT_CONFIG: dict[str, Any] = {}
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")
with open(_DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
    DEFAULT_CONFIG = json.load(f)


DEFAULT_MODEL_CONFIG: dict[str, dict[str, Any]] = DEFAULT_CONFIG["model_configs"]
DEFAULT_RANSAC_CONFIG: dict[str, Any] = DEFAULT_CONFIG["ransac"]
RANSAC_METHOD_MAP = {
    "RANSAC": cv2.RANSAC,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "LMEDS": cv2.LMEDS,
    "RHO": cv2.RHO,
}

EmbeddingModelName: type = Literal["clip", "dino"]
MatcherModelName: type = Literal["lightglue", "loma"]
MatchMethod: type = Literal["keypoint_matching", "line_ocr"]
ImageEmbedding: type = np.ndarray[tuple[Any, ...], np.dtype[np.floating]]


@dataclass
class ClipConfig:
    image_size: int = 336
    model_name: str = "PE-Core-L-14-336"
    model_path: str | Path = "./models/PE-Core-L-14-336/open_clip_pytorch_model.bin"
    top_k: int = 3
    embeddings_path: str | Path | None = None
    annoy_index_path: str | Path | None = None


@dataclass
class DinoConfig:
    image_size: int = 224
    model_name: str = "dinov3-vitl16-pretrain-lvd1689m"
    model_path: str | Path = "./models/dinov3-vitl16-pretrain-lvd1689m"
    top_k: int = 3
    embeddings_path: str | Path | None = None
    annoy_index_path: str | Path | None = None


@dataclass
class LightglueConfig:
    max_width: int | None = 1024
    min_match_score: float = 0.5
    min_matches: int = 10
    score_method: Literal["sum", "mean"] = "mean"
    top_k_matches: int = 100
    depth_confidence: int = -1
    width_confidence: int = -1


# keypoints, descriptions, height, width from LoMa detect-and-describe.
LomaImageFeatures = tuple[torch.Tensor, torch.Tensor, int, int]


@dataclass
class LomaConfig:
    model_name: str = "loma_B"
    min_match_score: float = 0.5
    min_matches: int = 10
    score_method: Literal["sum", "mean"] = "mean"
    top_k_matches: int = 100
    max_width: int | None = 1024
    cache_dir: str | Path | None = None
    cache_features: bool = False


@dataclass
class RfDetrConfig:
    model_path: str | Path = "./models/rf_detr_seg_nano_rulers.pth"
    detection_threshold: float = 0.5


@dataclass
class RansacConfig:
    ransac_iters: int = 100000
    ransac_method: str = "USAC_MAGSAC"
    reproj_threshold: float = 0.5
    confidence_threshold: float = 0.999

    @property
    def cv_method(self) -> int:
        if isinstance(self.ransac_method, int):
            return int(self.ransac_method)
        key = str(self.ransac_method).upper()
        if key not in RANSAC_METHOD_MAP:
            raise ValueError(
                f"Unknown RANSAC method: {self.ransac_method!r}. "
                f"Use one of {sorted(RANSAC_METHOD_MAP)}"
            )
        return RANSAC_METHOD_MAP[key]


@dataclass
class TickLabel:
    label: str
    shape_type: str
    orig_points: list[list[float, float]]
    normalized_points: list[list[float, float]]
    crop_points: list[list[float, float]]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        orig_points = self.orig_points
        if isinstance(orig_points, np.ndarray):
            orig_points = orig_points.tolist()

        normalized_points = self.normalized_points
        if isinstance(normalized_points, np.ndarray):
            normalized_points = normalized_points.tolist()
        crop_points = self.crop_points
        if isinstance(crop_points, np.ndarray):
            crop_points = crop_points.tolist()

        return {
            "label": self.label,
            "shape_type": self.shape_type,
            "orig_points": orig_points,
            "normalized_points": normalized_points,
            "crop_points": crop_points,
        }


@dataclass
class RulerMatch:
    matched_ruler_fname: str
    matched_ruler_path: str
    image_similarity: float
    overall_score: float
    num_keypoint_matches: int
    rotated: bool
    scale_factor: float
    scale_factor_std: float
    tick_labels: list[TickLabel]
    unit: str = DEFAULT_SI_UNIT
    method: MatchMethod = "keypoint_matching"
    ruler_bbox_xyxy: list[int] | None = None
    coco_annotation_id: Any = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_ruler_fname": self.matched_ruler_fname,
            "matched_ruler_path": self.matched_ruler_path,
            "image_similarity": self.image_similarity,
            "overall_score": self.overall_score,
            "num_keypoint_matches": self.num_keypoint_matches,
            "rotated": self.rotated,
            "scale_factor": self.scale_factor,
            "scale_factor_std": self.scale_factor_std,
            "tick_labels": [t.to_dict() for t in self.tick_labels],
            "scale_factor_unit": self.unit,
            "method": self.method,
            "ruler_bbox_xyxy": self.ruler_bbox_xyxy,
            "coco_annotation_id": self.coco_annotation_id,
        }


@dataclass
class DetectionImage:
    id: int
    file_name: str
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class DetectionCategory:
    id: int
    name: str
    supercategory: str = "object"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


@dataclass
class DetectionAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    segmentation: list[list[float]] = field(default_factory=list)
    iscrowd: int = 0
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "segmentation": self.segmentation,
            "area": self.area,
            "bbox": self.bbox,
            "iscrowd": self.iscrowd,
            "score": self.score,
        }


# Below follows the COCO format for object detection
# See https://cocodataset.org/#format-data
@dataclass
class DetectionResults:
    images: list[DetectionImage] = field(default_factory=list)
    annotations: list[DetectionAnnotation] = field(default_factory=list)
    categories: list[DetectionCategory] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [image.to_dict() for image in self.images],
            "annotations": [ann.to_dict() for ann in self.annotations],
            "categories": [category.to_dict() for category in self.categories],
        }

    @classmethod
    def from_coco(cls, coco: dict[str, Any]) -> DetectionResults:
        images = [
            DetectionImage(
                id=int(image["id"]),
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
            )
            for image in coco.get("images") or []
        ]
        annotations = [
            DetectionAnnotation(
                id=int(ann["id"]),
                image_id=int(ann["image_id"]),
                category_id=int(ann.get("category_id", 0)),
                segmentation=list(ann.get("segmentation") or []),
                area=float(ann.get("area", 0.0)),
                bbox=[float(value) for value in ann.get("bbox") or []],
                iscrowd=int(ann.get("iscrowd", 0)),
                score=float(ann.get("score", 1.0)),
            )
            for ann in coco.get("annotations") or []
        ]
        categories = [
            DetectionCategory(
                id=int(category["id"]),
                name=str(category["name"]),
                supercategory=str(category.get("supercategory", "object")),
            )
            for category in coco.get("categories") or []
        ]
        return cls(images=images, annotations=annotations, categories=categories)


@dataclass
class InferenceResult:
    status: str
    scale_factor: float | None = None
    scale_factor_std: float | None = None
    match_results: list[RulerMatch] = field(default_factory=list)
    detection_results: DetectionResults = field(default_factory=DetectionResults)
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scale_factor": self.scale_factor,
            "scale_factor_std": self.scale_factor_std,
            "match_results": [m.to_dict() for m in self.match_results],
            "detection_results": self.detection_results.to_dict(),
            "metadata": self.metadata,
            "message": self.message,
        }


class SquarePad:
    def __call__(self, img: RGBImage | Image.Image) -> Image.Image:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        w, h = img.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(img, padding, 0, "constant")


def _tick_label_from_dict(data: dict[str, Any]) -> TickLabel:
    payload = dict(data)
    if "crop_points" not in payload:
        payload["crop_points"] = payload.get("normalized_points", [])
    if "orig_points" not in payload:
        payload["orig_points"] = payload.get("normalized_points", [])
    if "shape_type" not in payload:
        payload["shape_type"] = "LineString"
    return _dataclass_from_dict(TickLabel, payload)


def _dataclass_from_dict(cls, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def _normalize_embedding_model(name: str) -> EmbeddingModelName:
    n = str(name).strip().lower()
    if "clip" in n:
        return "clip"
    elif "dino" in n:
        return "dino"
    else:
        raise ValueError(
            f"Unknown embedding model: {name}. Supported models are: {EmbeddingModelName}"
        )


def _normalize_matcher_model(name: str) -> MatcherModelName:
    n = str(name).strip().lower()
    if "lightglue" in n:
        return "lightglue"
    elif "loma" in n:
        return "loma"
    else:
        raise ValueError(
            f"Unknown matcher model: {name}. Supported models are: {MatcherModelName}"
        )


def _merge_model_configs(
    model_configs: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    merged = {
        section: {**DEFAULT_MODEL_CONFIG[section]} for section in DEFAULT_MODEL_CONFIG
    }
    if not model_configs:
        return merged

    for section in DEFAULT_MODEL_CONFIG:
        if section in model_configs:
            merged[section].update(model_configs[section])
    return merged


def _merge_ransac_config(ransac: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_RANSAC_CONFIG}
    if ransac:
        merged.update(ransac)
    return merged


def _validate_numpy_image(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Input image must be a 3-channel RGB image, got {img.shape}")
    elif img.dtype != np.uint8:
        raise ValueError(f"Input image must be a uint8 image, got {img.dtype}")


def _to_pil_image(input_img: str | Path | RGBImage | Image.Image) -> Image.Image:
    if isinstance(input_img, str | Path):
        return Image.open(input_img).convert("RGB")
    elif isinstance(input_img, Image.Image):
        return input_img.convert("RGB")
    elif isinstance(input_img, np.ndarray):
        _validate_numpy_image(input_img)
        return Image.fromarray(input_img)
    else:
        raise TypeError(f"Invalid input type: {type(input_img)}")


def _load_rgb(input_img: str | Path | RGBImage | Image.Image) -> RGBImage:
    if isinstance(input_img, str | Path):
        img = load_rgb_image(input_img)
        if img is None:
            raise ValueError(f"Could not read image: {input_img}")
        return img
    elif isinstance(input_img, Image.Image):
        return np.array(input_img.convert("RGB"), dtype=np.uint8)
    elif isinstance(input_img, np.ndarray):
        _validate_numpy_image(input_img)
        return input_img
    else:
        raise TypeError(f"Invalid input type: {type(input_img)}")


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def _pad_image_for_loma(
    img: RGBImage,
    resize: int,
    multiple: int = LOMA_SIZE_MULTIPLE,
) -> RGBImage:
    """Pad on the bottom and/or right so LoMa can downsample the image.

    After the detector's aspect-preserving resize (long side → ``resize``), both
    sides must be at least ``multiple`` and divisible by ``multiple``. Padding
    keeps original content in the top-left so pixel coords are unchanged.
    """
    h, w = img.shape[:2]
    target_h = _ceil_to_multiple(max(h, multiple), multiple)
    target_w = _ceil_to_multiple(max(w, multiple), multiple)

    long_side = max(target_w, target_h)
    short_min = _ceil_to_multiple(
        (multiple * long_side + resize - 1) // resize,
        multiple,
    )
    if target_w >= target_h:
        target_h = max(target_h, short_min)
    else:
        target_w = max(target_w, short_min)

    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)


def _prepare_query_image(
    query_img: RGBImage,
    max_width: int | None = 1024,
) -> tuple[RGBImage, RGBImage, tuple[int, int]]:
    """
    Prepare the query image for matching.
    Ensure longer side is horizontal and resize if necessary.
    Args:
        query_img: The query image to prepare
        max_width: The maximum width of the query image
    Returns:
        tuple[RGBImage, RGBImage, tuple[int, int]]: The prepared query image, the 180°-rotated query image, and the original shape of the query image

    """
    orig_shape = query_img.shape[:2]
    query_img = ensure_horizontal_image(query_img)

    h, w = query_img.shape[:2]
    if max_width is not None and w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = round(h * scale)
        query_img = cv2.resize(query_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    query_rot = cv2.rotate(query_img, cv2.ROTATE_180)
    return query_img, query_rot, orig_shape


def mm_pixel_dist_to_scale_factor(
    avg_dist_mm: float,
    std_dist_mm: float,
    target_unit: str,
) -> tuple[float, float] | None:
    """Convert px/mm distance to a scale factor in ``target_unit``.

    Returns ``(scale_factor, scale_factor_std)`` or ``None`` if invalid.
    """
    if avg_dist_mm > MAX_PIXEL_DIST_MM or avg_dist_mm < MIN_PIXEL_DIST_MM:
        logger.warning(
            "Skipping match with invalid pixel distance: avg_dist_mm={}, std_dist_mm={}",
            avg_dist_mm,
            std_dist_mm,
        )
        return None

    converted_pixel_dist = convert_unit(avg_dist_mm, unit="mm", target_unit=target_unit)
    scale_factor = 1 / converted_pixel_dist

    if std_dist_mm == 0:
        scale_factor_std = 0.0
    else:
        converted_pixel_dist_std = convert_unit(
            std_dist_mm, unit="mm", target_unit=target_unit
        )
        scale_factor_std = 1 / converted_pixel_dist_std

    return scale_factor, scale_factor_std


def filter_transformed_points(
    transformed_points: list[np.ndarray],
    gt_tick_labels: list[Any],
    image_shape: tuple[int, ...],
    strict: bool = False,
) -> tuple[list[np.ndarray], list[Any]]:
    """Keep transformed tick-mark groups that lie inside the query crop.

    Args:
        transformed_points: list of (N, 2) arrays, one per tick-label group
        gt_tick_labels: tick-label dicts aligned with transformed_points
        image_shape: query image shape (H, W, ...) or (H, W)
        strict: if True, keep a group only if every point is inside the crop.
            if False, keep a group if at least 2 points lie inside a 5px / 5% margin.

    Returns:
        filtered_points, filtered_labels (possibly empty)
    """
    h, w = image_shape[:2]
    if not strict:
        dx = min(5, 0.05 * w)
        dy = min(5, 0.05 * h)

    filtered_points: list[np.ndarray] = []
    filtered_labels: list[Any] = []
    for trans_pts, gt_tick_label in zip(transformed_points, gt_tick_labels):
        if trans_pts is None or len(trans_pts) < 2:
            continue

        if strict:
            x_min, y_min = trans_pts.min(axis=0)
            x_max, y_max = trans_pts.max(axis=0)
            if (x_min < 0) or (y_min < 0) or (x_max > w) or (y_max > h):
                continue
        else:
            xs, ys = trans_pts[:, 0], trans_pts[:, 1]
            inside = (-dx <= xs) & (xs <= w + dx) & (-dy <= ys) & (ys <= h + dy)
            if int(inside.sum()) < 2:
                continue

        filtered_points.append(trans_pts)
        filtered_labels.append(gt_tick_label)

    return filtered_points, filtered_labels


def _tick() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _elapsed_ms(start: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def _force_loma_fp32(matcher: torch.nn.Module) -> None:
    """Run DaD and the DINOv2 descriptor in float32.

    Those modules ignore LoMa's ``mp`` flag. They always autocast to the dtype
    chosen at import, which is float16 on GPUs below compute capability 8.
    Maxwell and Pascal run that dtype far slower than float32.
    """
    for module in matcher.modules():
        if hasattr(module, "amp"):
            module.amp = False
        if hasattr(module, "amp_dtype"):
            module.amp_dtype = torch.float32
        vit = getattr(module, "dinov2_vitl14", None)
        if vit is not None:
            vit.float()


def _set_model_cache_env(base_dir: Path, *, log: bool = True) -> None:
    """Keep LoMa and PaddleOCR weights under ``base_dir/models``.

    ``setdefault`` leaves an existing TORCH_HOME or PADDLE_PDX_CACHE_HOME in place.
    PaddleX reads its cache directory at import time, so this must run before
    ``paddleocr`` is imported.
    """
    cache_dir = base_dir / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if log:
        logger.info(
            "Model cache: TORCH_HOME={} | PADDLE_PDX_CACHE_HOME={}",
            os.environ["TORCH_HOME"],
            os.environ["PADDLE_PDX_CACHE_HOME"],
        )


class RulerInference:
    def __init__(
        self,
        base_dir: str | Path,
        device: str | None = None,
        model_configs: dict[str, Any] | None = None,
        target_unit: str = DEFAULT_SI_UNIT,
        strict: bool = bool(DEFAULT_CONFIG["strict"]),
        reference_dir: str | Path | None = None,
        embedding_model: str = str(DEFAULT_CONFIG["embedding_model"]),
        matcher_model: str = str(DEFAULT_CONFIG["matcher_model"]),
        ransac: dict[str, Any] | None = None,
    ):
        logger.info("Initializing RulerInference")
        self.base_dir = Path(base_dir)
        _set_model_cache_env(self.base_dir)
        self.reference_dir = (
            Path(reference_dir)
            if reference_dir is not None
            else self._resolve_reference_dir()
        )
        self.target_unit = target_unit
        self.strict = strict
        self.embedding_model_name = _normalize_embedding_model(embedding_model)
        self.matcher_model_name = _normalize_matcher_model(matcher_model)

        merged_configs = _merge_model_configs(model_configs)
        self.clip_config = _dataclass_from_dict(ClipConfig, merged_configs["clip"])
        self.dino_config = _dataclass_from_dict(DinoConfig, merged_configs["dino"])
        self.lightglue_config = _dataclass_from_dict(
            LightglueConfig, merged_configs["lightglue"]
        )
        self.loma_config = _dataclass_from_dict(LomaConfig, merged_configs["loma"])
        self.rf_detr_config = _dataclass_from_dict(
            RfDetrConfig, merged_configs["rf_detr"]
        )
        self.ransac_config = _dataclass_from_dict(
            RansacConfig, _merge_ransac_config(ransac)
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embeddings_path = self._resolve_embeddings_path()
        self.annoy_index_path = self._resolve_annoy_index_path()

        logger.info(
            "Config: base_dir={} | reference_dir={} | device={} | "
            "target_unit={} | strict={} | embedding={} | matcher={}",
            self.base_dir,
            self.reference_dir,
            self.device,
            self.target_unit,
            self.strict,
            self.embedding_model_name,
            self.matcher_model_name,
        )
        emb_cfg = self._embedding_config
        match_cfg = self._matcher_config
        logger.info(
            "{}: {} (top_k={}) | {}: min_matches={} top_k_matches={} | "
            "RANSAC: method={} iters={} reproj={} conf={}",
            self.embedding_model_name,
            emb_cfg.model_name,
            emb_cfg.top_k,
            self.matcher_model_name,
            match_cfg.min_matches,
            match_cfg.top_k_matches,
            self.ransac_config.ransac_method,
            self.ransac_config.ransac_iters,
            self.ransac_config.reproj_threshold,
            self.ransac_config.confidence_threshold,
        )
        # ImageNet preprocessing mean and standard deviations
        IMGNET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
        IMGNET_STD = np.array([0.26862954, 0.26130258, 0.27577711])
        self.custom_preprocess = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(
                    size=(emb_cfg.image_size, emb_cfg.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
            ]
        )

        self.clip_model = None
        self.dino_model = None
        self.extractor_model = None
        self.lightglue_matcher = None
        self.loma_matcher = None
        self.detection_threshold = float(self.rf_detr_config.detection_threshold)
        self.detection_model_path = self._resolve_detection_model_path(
            self.rf_detr_config.model_path
        )
        self._detector = None

        self._load_models()
        self._load_index()
        logger.info("RulerInference ready")

    @property
    def _embedding_config(self) -> ClipConfig | DinoConfig:
        if self.embedding_model_name == "dino":
            return self.dino_config
        elif self.embedding_model_name == "clip":
            return self.clip_config
        else:
            raise ValueError(
                f"Invalid embedding model: {self.embedding_model_name}, supported models: dino, clip"
            )

    @property
    def _matcher_config(self) -> LightglueConfig | LomaConfig:
        if self.matcher_model_name == "loma":
            return self.loma_config
        elif self.matcher_model_name == "lightglue":
            return self.lightglue_config
        else:
            raise ValueError(
                f"Invalid matcher model: {self.matcher_model_name}, supported models: loma, lightglue"
            )

    def _model_metadata(self) -> dict[str, str]:
        if self.matcher_model_name == "loma":
            matcher_name = "LoMa_B"
        else:
            matcher_name = "LightGlue"

        return {
            "embedding_model": self._embedding_config.model_name,
            "matcher_model": matcher_name,
        }

    def _resolve_reference_dir(self) -> Path:
        candidates = [
            self.base_dir / "dataset" / "merged_ruler_crops_normalized",
            self.base_dir / "dataset" / "ruler_crops_normalized",
        ]
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    def _resolve_detection_model_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        base_candidate = (self.base_dir / path).resolve()
        if base_candidate.exists():
            return base_candidate
        return path

    def _resolve_embeddings_path(self) -> Path:
        cfg = self._embedding_config
        if cfg.embeddings_path is not None:
            resolved = self._resolve_path(cfg.embeddings_path)
            if resolved is not None:
                return resolved
        if self.embedding_model_name == "dino":
            return (
                self.base_dir
                / "dataset"
                / f"embeddings_{self.dino_config.model_name}_custom.h5"
            )
        return self.base_dir / f"dataset/embeddings_{self.clip_config.model_name}.h5"

    def _resolve_annoy_index_path(self) -> Path:
        cfg = self._embedding_config
        if cfg.annoy_index_path is not None:
            resolved = self._resolve_path(cfg.annoy_index_path)
            if resolved is not None:
                return resolved
        if self.embedding_model_name == "dino":
            return self.base_dir / "dataset" / "dino_annoy_index_custom.ann"
        return self.base_dir / "dataset" / f"clip_{self.clip_config.model_name}.ann"

    def _load_models(self):
        if self.embedding_model_name == "dino":
            from transformers import AutoModel

            model_path = str(
                self._resolve_path(self.dino_config.model_path)
                or self.dino_config.model_path
            )
            logger.info("Loading DINO model from {}", model_path)
            self.dino_model = AutoModel.from_pretrained(model_path)
            self.dino_model.eval().to(self.device)
        else:
            import open_clip

            model_path = str(
                self._resolve_path(self.clip_config.model_path)
                or self.clip_config.model_path
            )
            logger.info(
                "Loading CLIP model {} from {}",
                self.clip_config.model_name,
                model_path,
            )
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                self.clip_config.model_name,
                pretrained=model_path,
            )
            self.clip_model.eval().to(self.device)

        if self.matcher_model_name == "loma":
            from loma import LoMa
            from loma.loma import LoMaB, LoMaR

            loma_name = self.loma_config.model_name
            filter_threshold = self.loma_config.min_match_score
            loma_kwargs: dict[str, Any] = {"filter_threshold": filter_threshold}
            # LoMa enables float16 autocast on every GPU below compute 8.0.
            # Maxwell (Titan X, 5.2) runs fp16 much slower than fp32.
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
                loma_kwargs["mp"] = False
                logger.info(
                    "Disabling LoMa mixed precision on compute capability {}",
                    torch.cuda.get_device_capability(),
                )
            logger.info(
                "Loading LoMa matcher {} (filter_threshold={}, mp={})",
                loma_name,
                filter_threshold,
                loma_kwargs.get("mp", True),
            )
            if loma_name.lower() == "loma_b":
                loma_cfg = LoMaB(**loma_kwargs)
            elif loma_name.lower() == "loma_r":
                loma_cfg = LoMaR(**loma_kwargs)
            else:
                raise ValueError(
                    f"Invalid LoMa model: {loma_name}. Use 'loma_B' or 'loma_R'."
                )
            self.loma_matcher = LoMa(loma_cfg)
            self.loma_matcher.eval()
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
                _force_loma_fp32(self.loma_matcher)
                logger.info(
                    "LoMa detector and DINOv2 descriptor forced to float32 on compute capability {}",
                    torch.cuda.get_device_capability(),
                )
        else:
            from lightglue import LightGlue, SuperPoint

            logger.info("Loading SuperPoint extractor and LightGlue matcher")
            self.extractor_model = (
                SuperPoint(max_num_keypoints=None).eval().to(self.device)
            )
            self.lightglue_matcher = (
                LightGlue(
                    features="superpoint",
                    depth_confidence=self.lightglue_config.depth_confidence,
                    width_confidence=self.lightglue_config.width_confidence,
                    filter_threshold=self.lightglue_config.min_match_score,
                )
                .eval()
                .to(self.device)
            )

    def _load_index(self):
        logger.info("Loading reference embeddings from {}", self.embeddings_path)
        with h5py.File(self.embeddings_path, "r") as h5:
            self.filenames = np.array([f.decode("utf-8") for f in h5["filenames"]])
            embeddings = np.array(h5["embeddings"], dtype=np.float32)

        logger.info(
            "Loaded {} reference embeddings (dim={})",
            len(self.filenames),
            embeddings.shape[-1],
        )

        self.annoy_index = AnnoyIndex(embeddings.shape[-1], "dot")
        if self.annoy_index_path.exists():
            logger.info("Loading Annoy index from {}", self.annoy_index_path)
            self.annoy_index.load(str(self.annoy_index_path))
        else:
            logger.warning(
                "Annoy index not found at {}; building and saving",
                self.annoy_index_path,
            )
            for idx, embedding in enumerate(embeddings):
                self.annoy_index.add_item(idx, embedding)
            self.annoy_index.build(10)
            self.annoy_index.save(str(self.annoy_index_path))
            logger.info("Saved Annoy index to {}", self.annoy_index_path)

    def get_clip_embedding(
        self, input_img: str | Path | RGBImage | Image.Image
    ) -> ImageEmbedding:
        if self.clip_model is None:
            raise RuntimeError(
                "CLIP model is not loaded (embedding_model is not 'clip')"
            )
        img = (
            self.custom_preprocess(_to_pil_image(input_img))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode(), torch.amp.autocast(self.device):
            img_features = self.clip_model.encode_image(img, normalize=True)
            img_features = img_features[0]
        return img_features.cpu().numpy()

    def get_dino_embedding(
        self, input_img: str | Path | RGBImage | Image.Image
    ) -> ImageEmbedding:
        if self.dino_model is None:
            raise RuntimeError(
                "DINO model is not loaded (embedding_model is not 'dino')"
            )
        img = _to_pil_image(input_img)
        pixel_values = self.custom_preprocess(img).unsqueeze(0).to(self.device)
        inputs = {"pixel_values": pixel_values}
        started = _tick()
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
        logger.debug("DINO embedding took {:.1f} ms", _elapsed_ms(started))
        embedding = outputs.pooler_output.cpu().numpy().squeeze()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def get_embedding(
        self, input_img: str | Path | RGBImage | Image.Image
    ) -> ImageEmbedding:
        """Embed an image with the currently selected embedding model (CLIP or DINO)."""
        if self.embedding_model_name == "dino":
            return self.get_dino_embedding(input_img)
        return self.get_clip_embedding(input_img)

    def _load_reference_tick_data(
        self, reference_path: str
    ) -> tuple[list[dict[str, Any]], list[np.ndarray], list[str]] | None:
        gt_ruler_data_path = reference_path.replace(".jpg", ".json")
        with open(gt_ruler_data_path, "r", encoding="utf-8") as f:
            gt_ruler_data = json.load(f)

        gt_tick_labels = gt_ruler_data.get("tick_labels", [])
        if not gt_tick_labels:
            return None

        tick_points = [
            np.array(tick_label["normalized_points"], dtype=np.float32)
            for tick_label in gt_tick_labels
        ]
        tick_label_names = [tick_label["label"] for tick_label in gt_tick_labels]
        return gt_tick_labels, tick_points, tick_label_names

    def extract_match_keypoints_lightglue(
        self,
        img0: str | Path | RGBImage,
        img1: str | Path | RGBImage,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match keypoints with LightGlue. Returns kpts0, kpts1, matches (N, 2), match_scores."""
        from lightglue.utils import rbd

        if self.extractor_model is None or self.lightglue_matcher is None:
            raise RuntimeError(
                "LightGlue is not loaded (matcher_model is not 'lightglue')"
            )
        img0_np = _load_rgb(img0)
        img1_np = _load_rgb(img1)
        with torch.inference_mode():
            feats0 = self.extractor_model.extract(
                numpy_image_to_torch(img0_np).to(self.device)
            )
            feats1 = self.extractor_model.extract(
                numpy_image_to_torch(img1_np).to(self.device)
            )
            matches01 = self.lightglue_matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0 = feats0["keypoints"].cpu().numpy()
        kpts1 = feats1["keypoints"].cpu().numpy()
        matches = matches01["matches"].cpu().numpy()
        match_scores = matches01["scores"].cpu().numpy()
        return kpts0, kpts1, matches, match_scores

    def _loma_cache_path(self, img_path: str | Path) -> Path:
        loma_name = self.loma_config.model_name
        cache_dir = self._resolve_path(self.loma_config.cache_dir)
        if cache_dir is None:
            cache_dir = self.base_dir / "dataset" / f"{loma_name}_features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{Path(img_path).stem}.p"

    def _numpy_to_loma_tensor(
        self, img: RGBImage, width: int, height: int
    ) -> torch.Tensor:
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return numpy_image_to_torch(resized).unsqueeze(0).to(self.device)

    def _loma_detect_and_describe_array(self, img: RGBImage) -> LomaImageFeatures:
        """
        Detect & describe an in-memory image crop using LoMa
        LoMa uses two models, which take diferent input sizes.
        This function resizes the image to the appropriate size for the detector and descriptor.
        """
        if self.loma_matcher is None:
            raise RuntimeError("LoMa is not loaded (matcher_model is not 'loma')")

        detector = self.loma_matcher._detector
        descriptor = self.loma_matcher._descriptor
        num_keypoints = self.loma_matcher.cfg.num_keypoints

        img = _pad_image_for_loma(img, detector.resize)
        orig_h, orig_w = img.shape[:2]
        multiple = LOMA_SIZE_MULTIPLE

        if detector.keep_aspect_ratio:
            scale = detector.resize / max(orig_w, orig_h)
            det_w = max(multiple, int((scale * orig_w) // multiple * multiple))
            det_h = max(multiple, int((scale * orig_h) // multiple * multiple))
        else:
            det_w = det_h = detector.resize
        det_tensor = self._numpy_to_loma_tensor(img, det_w, det_h)
        started = _tick()
        keypoints = detector.detect(det_tensor, num_keypoints=num_keypoints)[
            "keypoints"
        ]
        logger.debug(
            "LoMa detect {}x{} took {:.1f} ms",
            det_h,
            det_w,
            _elapsed_ms(started),
        )

        # DeDoDeDescriptor.read_image / describe_keypoints_from_path defaults
        desc_tensor = self._numpy_to_loma_tensor(img, 784, 784)
        started = _tick()
        descriptions = descriptor.describe_keypoints(desc_tensor, keypoints)[
            "descriptions"
        ]
        logger.debug("LoMa describe 784x784 took {:.1f} ms", _elapsed_ms(started))
        return keypoints, descriptions, orig_h, orig_w

    def _loma_detect_and_describe(
        self, img: str | Path | RGBImage, cache_features: bool = False
    ) -> LomaImageFeatures:
        if self.loma_matcher is None:
            raise RuntimeError("LoMa is not loaded (matcher_model is not 'loma')")

        if isinstance(img, str | Path):
            cache_path = self._loma_cache_path(img) if cache_features else None
            if cache_path is not None and cache_path.exists():
                started = _tick()
                with open(cache_path, "rb") as f:
                    kpts, desc, h, w = pickle.load(f)
                logger.debug(
                    "LoMa feature cache hit {} took {:.1f} ms",
                    Path(img).name,
                    _elapsed_ms(started),
                )
                return (
                    kpts.to(self.device),
                    desc.to(self.device),
                    h,
                    w,
                )

            started = _tick()
            packed = self.loma_matcher.detect_and_describe(str(img))
            logger.debug(
                "LoMa detect_and_describe {} took {:.1f} ms",
                Path(img).name,
                _elapsed_ms(started),
            )
            if cache_path is not None:
                kpts, desc, h, w = packed
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        (kpts.detach().cpu(), desc.detach().cpu(), int(h), int(w)),
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            return packed

        img_np = img if isinstance(img, np.ndarray) else _load_rgb(img)
        return self._loma_detect_and_describe_array(img_np)

    def extract_match_keypoints_loma(
        self,
        img0: str | Path | RGBImage,
        img1: str | Path | RGBImage,
        filter_threshold: float | None = None,
        feats0: LomaImageFeatures | None = None,
        feats1: LomaImageFeatures | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match keypoints with LoMa. Same return format as extract_match_keypoints_lightglue:
        kpts0, kpts1 (pixel coords), matches (N, 2 index pairs), match_scores.

        ``feats0`` / ``feats1`` are optional ``(keypoints, descriptions, height, width)``
        packs from ``_loma_detect_and_describe``. Pass them to skip detection when the
        same image is matched more than once.
        """
        from loma.loma import filter_matches, to_pixel_coords

        if self.loma_matcher is None:
            raise RuntimeError("LoMa is not loaded (matcher_model is not 'loma')")

        cache_features = self.loma_config.cache_features
        if feats0 is None:
            kpts0, desc0, h0, w0 = self._loma_detect_and_describe(
                img0, cache_features=cache_features
            )
        else:
            kpts0, desc0, h0, w0 = feats0
        if feats1 is None:
            kpts1, desc1, h1, w1 = self._loma_detect_and_describe(
                img1, cache_features=cache_features
            )
        else:
            kpts1, desc1, h1, w1 = feats1

        started = _tick()
        with torch.inference_mode():
            scores = self.loma_matcher(kpts0, kpts1, desc0, desc1)["scores"]
        logger.debug("LoMa match forward took {:.1f} ms", _elapsed_ms(started))

        if filter_threshold is None:
            filter_threshold = self.loma_config.min_match_score

        m0, _, mscores0, _ = filter_matches(scores, filter_threshold)
        valid = m0[0] > -1
        idx0 = torch.where(valid)[0]
        idx1 = m0[0][valid]

        kpts0_px = to_pixel_coords(kpts0[0], h0, w0).cpu().numpy()
        kpts1_px = to_pixel_coords(kpts1[0], h1, w1).cpu().numpy()
        matches = torch.stack([idx0, idx1], dim=-1).cpu().numpy()
        match_scores = mscores0[0][valid].cpu().numpy()
        return kpts0_px, kpts1_px, matches, match_scores

    def extract_match_keypoints(
        self,
        img0: str | Path | RGBImage,
        img1: str | Path | RGBImage,
        feats0: LomaImageFeatures | None = None,
        feats1: LomaImageFeatures | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match keypoints with the currently selected matcher (LightGlue or LoMa).

        Returns kpts0, kpts1, matches (N, 2), match_scores.
        ``feats0`` / ``feats1`` are LoMa feature packs and are ignored by LightGlue.
        """
        if self.matcher_model_name == "loma":
            return self.extract_match_keypoints_loma(
                img0, img1, feats0=feats0, feats1=feats1
            )
        return self.extract_match_keypoints_lightglue(img0, img1)

    def _run_matching(
        self,
        query_img: RGBImage,
        target_images: list[RGBImage],
        target_tick_points_list: list[list[np.ndarray]],
        target_tick_labels_list: list[list[str]],
        target_paths: list[str] | None = None,
        target_names: list[str] | None = None,
    ) -> list[dict[str, Any] | None]:

        match_cfg = self._matcher_config
        min_matches = match_cfg.min_matches
        top_k_matches = match_cfg.top_k_matches
        score_method = match_cfg.score_method
        ransac = self.ransac_config

        query_img, query_img_rot, orig_query_img_shape = _prepare_query_image(
            query_img, max_width=match_cfg.max_width
        )
        # Query crops are numpy arrays, so the on-disk LoMa cache (path stems only)
        # never applies. Detect each orientation once and reuse it for every target.
        query_feats: LomaImageFeatures | None = None
        query_feats_rot: LomaImageFeatures | None = None
        if self.matcher_model_name == "loma":
            query_feats = self._loma_detect_and_describe(query_img)
            query_feats_rot = self._loma_detect_and_describe(query_img_rot)
        match_results: list[dict[str, Any] | None] = []

        if target_paths is None:
            target_inputs: list[str | RGBImage] = list(target_images)
        else:
            target_inputs = list(target_paths)

        for i, (target_input, target_tick_points, target_tick_labels) in enumerate(
            zip(target_inputs, target_tick_points_list, target_tick_labels_list)
        ):
            if target_names is not None and i < len(target_names):
                target_name = target_names[i]
            elif isinstance(target_input, str | Path):
                target_name = Path(target_input).name
            else:
                target_name = f"candidate {i}"
            rotated = False
            target_feats: LomaImageFeatures | None = None
            if self.matcher_model_name == "loma":
                target_feats = self._loma_detect_and_describe(
                    target_input, cache_features=self.loma_config.cache_features
                )
            kpts0, kpts1, matches, match_scores = self.extract_match_keypoints(
                query_img,
                target_input,
                feats0=query_feats,
                feats1=target_feats,
            )
            num_matches = len(matches)

            kpts0_rot, kpts1_rot, matches_rot, match_scores_rot = (
                self.extract_match_keypoints(
                    query_img_rot,
                    target_input,
                    feats0=query_feats_rot,
                    feats1=target_feats,
                )
            )
            num_matches_rot = len(matches_rot)

            if num_matches_rot > num_matches:
                kpts0, kpts1, matches, match_scores = (
                    kpts0_rot,
                    kpts1_rot,
                    matches_rot,
                    match_scores_rot,
                )
                num_matches = num_matches_rot
                rotated = True

            if num_matches < min_matches:
                logger.debug(
                    "No match for {}: {} keypoint matches (minimum {})",
                    target_name,
                    num_matches,
                    min_matches,
                )
                match_results.append(None)
                continue

            sorted_idx = np.argsort(-match_scores)
            matches = matches[sorted_idx]
            match_scores = match_scores[sorted_idx]

            top_matches = matches[:top_k_matches]
            if score_method == "sum":
                overall_score = float(match_scores[:top_k_matches].sum())
            else:
                overall_score = float(match_scores[:top_k_matches].mean())

            dst_pts = kpts0[top_matches[:, 0]]
            src_pts = kpts1[top_matches[:, 1]]
            started = _tick()
            H, _ = cv2.findHomography(
                src_pts,
                dst_pts,
                method=ransac.cv_method,
                maxIters=ransac.ransac_iters,
                confidence=ransac.confidence_threshold,
                ransacReprojThreshold=ransac.reproj_threshold,
            )
            logger.debug(
                "Homography for {} took {:.1f} ms",
                target_name,
                _elapsed_ms(started),
            )
            if H is None:
                logger.debug("No homography for {}", target_name)
                match_results.append(None)
                continue

            target_tick_points_transformed = [
                cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
                for pts in target_tick_points
            ]

            new_h, new_w = query_img.shape[:2]
            if rotated:
                target_tick_points_transformed = [
                    np.array([new_w - 1, new_h - 1], dtype=np.float32) - pts
                    for pts in target_tick_points_transformed
                ]

            orig_h, orig_w = orig_query_img_shape[:2]
            factor_w = orig_w / new_w
            factor_h = orig_h / new_h
            target_tick_points_transformed = [
                pts * np.array([factor_w, factor_h])
                for pts in target_tick_points_transformed
            ]

            pixel_dists = [
                calculate_pixel_distance(
                    pts, tick_label=tick_label, target_unit=self.target_unit
                )
                for pts, tick_label in zip(
                    target_tick_points_transformed, target_tick_labels
                )
            ]

            match_results.append(
                {
                    "overall_score": overall_score,
                    "rotated": rotated,
                    "avg_pixel_dist_cm": float(
                        np.mean([dist[0] for dist in pixel_dists])
                    ),
                    "std_pixel_dist_cm": float(
                        np.mean([dist[1] for dist in pixel_dists])
                    ),
                    "num_keypoint_matches": int(num_matches),
                    "predicted_tick_points": target_tick_points_transformed,
                    "predicted_tick_labels": target_tick_labels,
                }
            )

        return match_results

    def _try_line_ocr_match(
        self,
        query_img: RGBImage,
        clip_info: dict[str, Any],
        orig_query_img_shape: tuple[int, int],
    ) -> RulerMatch | None:
        """Estimate scale factor via line segmentation + OCR. Returns None on failure."""
        _set_model_cache_env(self.base_dir, log=False)
        from .OCR import predict_line_OCR

        fname = clip_info["matched_ruler_fname"]
        query_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
        predictions = predict_line_OCR(query_bgr)
        if not predictions:
            logger.debug("Line OCR found no measurement for {}", fname)
            return None

        pred = predictions[0]
        converted = mm_pixel_dist_to_scale_factor(
            float(pred["avg_dist"]),
            float(pred["std_dist"]),
            self.target_unit,
        )
        if converted is None:
            logger.debug("Line OCR measurement out of range for {}", fname)
            return None
        scale_factor, scale_factor_std = converted

        transformed_points = [
            np.array(pts, dtype=np.float32) for pts in pred["transformed_points"]
        ]
        gt_tick_labels = [dict(t) for t in pred["gt_tick_labels"]]
        filtered_points, filtered_labels = filter_transformed_points(
            transformed_points,
            gt_tick_labels,
            orig_query_img_shape,
            strict=self.strict,
        )
        if len(filtered_points) < 1:
            logger.debug("Line OCR tick points fell outside the crop for {}", fname)
            return None

        for tick_label, t_pts in zip(filtered_labels, filtered_points):
            tick_label["normalized_points"] = t_pts
            tick_label["crop_points"] = t_pts

        return RulerMatch(
            matched_ruler_fname=clip_info["matched_ruler_fname"],
            matched_ruler_path=clip_info["matched_ruler_path"],
            image_similarity=clip_info["clip_similarity"],
            overall_score=float(pred.get("overall_score") or 1.0),
            num_keypoint_matches=int(pred.get("num_matches") or 0),
            rotated=bool(pred.get("used_rotation", False)),
            scale_factor=scale_factor,
            scale_factor_std=scale_factor_std,
            tick_labels=[_tick_label_from_dict(t) for t in filtered_labels],
            unit=self.target_unit,
            method="line_ocr",
        )

    def _predict_crop(self, query_img: RGBImage) -> InferenceResult:
        query_img, _, _ = _prepare_query_image(query_img, max_width=None)
        orig_query_img_shape = query_img.shape[:2]

        metadata = self._model_metadata()
        metadata["query_img_shape"] = query_img.shape[:2]

        embedding = self.get_embedding(query_img)
        top_k = self._embedding_config.top_k
        best_matches, best_sims = self.annoy_index.get_nns_by_vector(
            embedding, top_k, include_distances=True
        )

        matcher_label = "LoMa" if self.matcher_model_name == "loma" else "LightGlue"
        embedding_label = "DINO" if self.embedding_model_name == "dino" else "CLIP"

        if best_matches:
            best_fname = str(self.filenames[best_matches[0]])
            best_ruler_type = get_ruler_type(best_fname)
            if "line" in best_ruler_type.lower():
                logger.debug(
                    "Using OCR + CV line method (matched ruler_type={})",
                    best_ruler_type,
                )
                ocr_match = self._try_line_ocr_match(
                    query_img,
                    {
                        "matched_ruler_fname": best_fname,
                        "matched_ruler_path": str(self.reference_dir / best_fname),
                        "clip_similarity": float(best_sims[0]),
                    },
                    orig_query_img_shape,
                )
                if ocr_match is not None:
                    return InferenceResult(
                        status="success",
                        scale_factor=ocr_match.scale_factor,
                        scale_factor_std=ocr_match.scale_factor_std,
                        match_results=[ocr_match],
                        metadata=metadata,
                    )
                else:
                    logger.warning(
                        "OCR line method failed for matched type {}; using keypoint matching",
                        best_ruler_type,
                    )

        target_images: list[RGBImage] = []
        target_paths: list[str] = []
        target_tick_points_list: list[list[np.ndarray]] = []
        target_tick_labels_list: list[list[str]] = []
        clip_candidates: list[dict[str, Any]] = []
        gt_tick_labels_by_idx: list[list[dict[str, Any]]] = []

        for idx, sim in zip(best_matches, best_sims):
            reference_fname = str(self.filenames[idx])
            reference_path = str(self.reference_dir / reference_fname)
            tick_data = self._load_reference_tick_data(reference_path)
            target_img = load_rgb_image(reference_path)
            if tick_data is None or target_img is None:
                missing = []
                if tick_data is None:
                    missing.append("tick labels")
                if target_img is None:
                    missing.append("image")
                logger.debug(
                    "Skipping reference {}: missing {}",
                    reference_fname,
                    " and ".join(missing),
                )
                continue

            gt_tick_labels, tick_points, tick_label_names = tick_data
            target_images.append(target_img)
            target_paths.append(reference_path)
            target_tick_points_list.append(tick_points)
            target_tick_labels_list.append(tick_label_names)
            gt_tick_labels_by_idx.append(gt_tick_labels)
            clip_candidates.append(
                {
                    "matched_ruler_fname": reference_fname,
                    "matched_ruler_path": reference_path,
                    "clip_similarity": float(sim),
                }
            )

        if not target_images:
            logger.debug(
                "No valid {} candidates with reference tick data",
                embedding_label,
            )
            return InferenceResult(
                status="no_match",
                metadata=metadata,
                message=f"No valid {embedding_label} candidates with reference tick data.",
            )

        match_kwargs = {
            "query_img": query_img,
            "target_images": target_images,
            "target_tick_points_list": target_tick_points_list,
            "target_tick_labels_list": target_tick_labels_list,
        }
        if self.matcher_model_name == "loma":
            match_kwargs["target_paths"] = target_paths
        match_kwargs["target_names"] = [
            c["matched_ruler_fname"] for c in clip_candidates
        ]
        match_results_raw = self._run_matching(**match_kwargs)

        match_results: list[RulerMatch] = []

        for clip_info, match_result, gt_tick_labels in zip(
            clip_candidates, match_results_raw, gt_tick_labels_by_idx
        ):
            if match_result is None:
                continue

            transformed_points = [
                np.array(pts, dtype=np.float32)
                for pts in match_result["predicted_tick_points"]
            ]
            transformed_tick_labels = [dict(t) for t in gt_tick_labels]
            filtered_points, filtered_labels = filter_transformed_points(
                transformed_points,
                transformed_tick_labels,
                orig_query_img_shape,
                strict=self.strict,
            )
            if len(filtered_points) < 1:
                logger.debug(
                    "No tick points inside the crop for {}",
                    clip_info["matched_ruler_fname"],
                )
                continue

            kept_dists: list[tuple[float, float]] = []
            kept_points: list[np.ndarray] = []
            kept_labels: list[dict[str, Any]] = []
            for pts, tick_label in zip(filtered_points, filtered_labels):
                avg_dist, std_dist = calculate_pixel_distance(
                    pts, tick_label=tick_label["label"], target_unit="mm"
                )
                if avg_dist == 0:
                    logger.info(
                        "Skipping tick label, average pixel distance is 0",
                    )
                    continue

                if std_dist != 0:
                    std_ratio = std_dist / avg_dist
                    if std_ratio > CF_STD_THRESH:
                        logger.info(
                            f"Skipping tick label, transformed scale factor std. deviation ratio too high: {std_ratio} > {CF_STD_THRESH}",
                        )
                        continue
                kept_dists.append((avg_dist, std_dist))
                kept_points.append(pts)
                kept_labels.append(tick_label)

            if not kept_dists:
                logger.debug(
                    "No usable tick labels for {}",
                    clip_info["matched_ruler_fname"],
                )
                continue

            avg_dist_mm = float(np.mean([d[0] for d in kept_dists]))
            std_dist_mm = float(np.mean([d[1] for d in kept_dists]))
            converted = mm_pixel_dist_to_scale_factor(
                avg_dist_mm, std_dist_mm, self.target_unit
            )
            if converted is None:
                logger.debug(
                    "Scale factor out of range for {}",
                    clip_info["matched_ruler_fname"],
                )
                continue
            scale_factor, scale_factor_std = converted

            for tick_label, t_pts in zip(kept_labels, kept_points):
                tick_label["normalized_points"] = t_pts
                tick_label["crop_points"] = t_pts

            tick_labels = [_tick_label_from_dict(t) for t in kept_labels]
            match_results.append(
                RulerMatch(
                    matched_ruler_fname=clip_info["matched_ruler_fname"],
                    matched_ruler_path=clip_info["matched_ruler_path"],
                    image_similarity=clip_info["clip_similarity"],
                    overall_score=match_result["overall_score"],
                    num_keypoint_matches=match_result["num_keypoint_matches"],
                    rotated=match_result["rotated"],
                    scale_factor=scale_factor,
                    scale_factor_std=scale_factor_std,
                    tick_labels=tick_labels,
                    unit=self.target_unit,
                    method="keypoint_matching",
                )
            )

        match_results.sort(key=lambda m: m.overall_score, reverse=True)

        if not match_results:
            logger.debug(
                "No valid {} match for {} candidates",
                matcher_label,
                embedding_label,
            )
            return InferenceResult(
                status="no_match",
                metadata=metadata,
                message=f"No valid {matcher_label} match found for {embedding_label} candidates.",
            )

        best_match = match_results[0]
        return InferenceResult(
            status="success",
            scale_factor=best_match.scale_factor,
            scale_factor_std=best_match.scale_factor_std,
            match_results=match_results,
            metadata=metadata,
        )

    def _attach_original_points(
        self,
        result: InferenceResult,
        bbox: tuple[int, int, int, int],
    ) -> InferenceResult:
        """Map crop tick coordinates onto the original image via the ruler bbox."""

        def _map_ticks(ticks: list[TickLabel]) -> None:
            for tick in ticks:
                pts = np.array(tick.normalized_points, dtype=float)
                orig = crop_points_to_original(pts, bbox)
                tick.orig_points = orig.tolist()
                if isinstance(tick.crop_points, np.ndarray):
                    tick.crop_points = tick.crop_points.tolist()
                if isinstance(tick.normalized_points, np.ndarray):
                    tick.normalized_points = tick.normalized_points.tolist()

        for match in result.match_results:
            _map_ticks(match.tick_labels)
        result.metadata["ruler_bbox_xyxy"] = list[int](bbox)
        return result

    def _ensure_detector(self) -> None:
        if self._detector is not None:
            return
        from rfdetr import RFDETRSegNano

        path = self.detection_model_path
        if not path.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {path}")
        logger.info("Loading RF-DETR ruler detector from {}", path)
        self._detector = RFDETRSegNano.from_checkpoint(str(path))

    def detect_rulers(
        self,
        image: str | Path | RGBImage | Image.Image,
        threshold: float | None = None,
    ):
        """Detect rulers on a herbarium sheet.

        ``image`` is a file path or an RGB image (NumPy or PIL). Returns the
        RF-DETR ``supervision.Detections`` result.
        """
        self._ensure_detector()
        if threshold is None:
            threshold = self.detection_threshold

        if isinstance(image, str | Path):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            model_input: str | np.ndarray = str(path)
        else:
            model_input = _load_rgb(image)

        started = _tick()
        raw = self._detector.predict(
            model_input,
            threshold=float(threshold),
            include_source_image=False,
        )
        logger.debug("RF-DETR detection took {:.1f} ms", _elapsed_ms(started))
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        n_rulers = 0 if raw is None else len(raw)
        logger.debug("Detected {} ruler(s)", n_rulers)
        if raw is None:
            import supervision as sv

            return sv.Detections.empty()
        return raw

    def predict(
        self,
        input_img: str | Path | RGBImage | Image.Image,
        prediction_path: str | Path | dict[str, Any] | None = None,
    ) -> InferenceResult:
        """Estimate the scale factor using the ruler(s) on a given image.

        Args:
            input_img: Image path, or an RGB image (NumPy or PIL).
            prediction_path: Optional COCO JSON path or dict of ruler polygons.
                When omitted, RF-DETR detects the rulers and those polygons are cropped.
        Returns:
            InferenceResult, containing the predicted scale factor and other metadata.
        """
        if isinstance(input_img, str | Path):
            query_img_path = Path(input_img)
            if not query_img_path.exists():
                raise FileNotFoundError(f"Query image not found: {query_img_path}")
            sheet = load_rgb_image(query_img_path)
            if sheet is None:
                raise ValueError(f"Could not read query image: {query_img_path}")
            fname = query_img_path.name
        else:
            sheet = _load_rgb(input_img)
            fname = "image"

        if prediction_path is None:
            height, width = sheet.shape[:2]
            coco = detections_to_coco(fname, width, height, self.detect_rulers(sheet))
        elif isinstance(prediction_path, str | Path):
            with open(prediction_path, encoding="utf-8") as f:
                coco = json.load(f)
        else:
            coco = prediction_path

        return self._predict_coco(sheet, coco)

    def _predict_coco(
        self,
        sheet: RGBImage,
        coco: dict[str, Any],
    ) -> InferenceResult:
        crops = crop_rulers_from_coco(sheet, coco)
        metadata = self._model_metadata()
        metadata["image_shape"] = sheet.shape[:2]
        detection_results = DetectionResults.from_coco(coco)
        if not crops:
            logger.debug("No ruler polygons found")
            return InferenceResult(
                status="no_match",
                detection_results=detection_results,
                metadata=metadata,
                message="No ruler polygons found.",
            )

        all_matches: list[RulerMatch] = []
        for crop, bbox, ann in crops:
            result = self._predict_crop(crop)
            result = self._attach_original_points(result, bbox)
            bbox_list = list[int](bbox)
            ann_id = ann.get("id")
            for match in result.match_results:
                match.ruler_bbox_xyxy = bbox_list
                match.coco_annotation_id = ann_id
                all_matches.append(match)

        all_matches.sort(key=lambda m: m.overall_score, reverse=True)

        if all_matches:
            best = all_matches[0]

            return InferenceResult(
                status="success",
                scale_factor=best.scale_factor,
                scale_factor_std=best.scale_factor_std,
                match_results=all_matches,
                detection_results=detection_results,
                metadata=metadata,
            )

        logger.debug("No valid match found for any detected ruler")
        return InferenceResult(
            status="no_match",
            detection_results=detection_results,
            metadata=metadata,
            message="No valid match found for any detected ruler.",
        )

    def predict_and_save(
        self,
        query_img_path: str | Path,
        output_path: str | Path | None = None,
        prediction_path: str | Path | None = None,
    ) -> Path:
        query_img_path = Path(query_img_path)
        result = self.predict(query_img_path, prediction_path=prediction_path)

        if output_path is None:
            output_path = query_img_path.with_suffix(".json")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        return output_path

    def predict_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """Run inference on all images in input_dir and write JSON files to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for img_path in _collect_images(input_dir):
            out_path = output_dir / f"{img_path.stem}.json"
            saved_paths.append(self.predict_and_save(img_path, out_path))
        return saved_paths


def _collect_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(
        p for p in path.iterdir() if p.suffix.lower() in extensions and p.is_file()
    )


def load_config(cfg_path: str | Path | None = None) -> dict[str, Any]:
    """Load RulerInference kwargs from a JSON config file, applying defaults for missing keys."""
    if cfg_path is None:
        return DEFAULT_CONFIG
    else:
        cfg_path = Path(cfg_path).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a JSON object: {cfg_path}")

    model_configs_raw = cfg.get("model_configs")
    if model_configs_raw is not None and not isinstance(model_configs_raw, dict):
        raise ValueError("Config key 'model_configs' must be an object")

    ransac_raw = cfg.get("ransac")
    if ransac_raw is not None and not isinstance(ransac_raw, dict):
        raise ValueError("Config key 'ransac' must be an object")

    target_unit = cfg.get("target_unit", DEFAULT_CONFIG["target_unit"])
    if not isinstance(target_unit, str):
        raise TypeError("Config key 'target_unit' must be a string")

    # base_dir is the project root (dataset, models, indexes). Resolve relative
    # paths against the process CWD, not the config file directory.
    base_dir = Path(cfg.get("base_dir", DEFAULT_CONFIG["base_dir"]))
    if not base_dir.is_absolute():
        base_dir = base_dir.resolve()

    engine_cfg: dict[str, Any] = {
        "base_dir": base_dir,
        "strict": bool(cfg.get("strict", DEFAULT_CONFIG["strict"])),
        "target_unit": target_unit,
        "model_configs": _merge_model_configs(model_configs_raw),
        "ransac": _merge_ransac_config(ransac_raw),
        "embedding_model": cfg.get(
            "embedding_model", DEFAULT_CONFIG["embedding_model"]
        ),
        "matcher_model": cfg.get("matcher_model", DEFAULT_CONFIG["matcher_model"]),
    }
    if cfg.get("device") is not None:
        engine_cfg["device"] = str(cfg["device"])

    return engine_cfg


def run_inference(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
    strict: bool = bool(DEFAULT_CONFIG["strict"]),
    model_configs: dict[str, Any] | None = None,
    target_unit: str = DEFAULT_SI_UNIT,
    device: str | None = None,
    embedding_model: str = str(DEFAULT_CONFIG["embedding_model"]),
    matcher_model: str = str(DEFAULT_CONFIG["matcher_model"]),
    ransac: dict[str, Any] | None = None,
) -> list[Path]:
    """Run ruler inference on all images in input_dir and save JSON results to output_dir."""
    engine = RulerInference(
        base_dir=base_dir,
        device=device,
        model_configs=model_configs,
        target_unit=target_unit,
        strict=strict,
        embedding_model=embedding_model,
        matcher_model=matcher_model,
        ransac=ransac,
    )
    return engine.predict_directory(input_dir, output_dir)
