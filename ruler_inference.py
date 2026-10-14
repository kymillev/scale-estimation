import json
import sys
from pathlib import Path

from loguru import logger
from scale_estimation.engine import RulerInference, load_config
from tqdm.auto import tqdm

# Default log level. Set to "DEBUG" to see more info.
logger.remove()
logger.add(sys.stderr, level="INFO")

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}

# Load default config
cfg = load_config()

# Load engine and models
engine = RulerInference(**cfg)

# Run inference on all images in the sample dataset
img_dir = Path("dataset") / "sample_images"
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in tqdm(img_dir.iterdir()):
    if img_path.suffix.lower() not in IMAGE_EXTS:
        continue
    if "19061_plantclef" not in img_path.name:
        continue
    out_path = output_dir / f"{img_path.stem}.json"

    result = engine.predict(img_path)

    best_match = result.match_results[0] if result.match_results else None
    detection_results = result.detection_results

    if len(detection_results.annotations) == 0:
        logger.warning(f"No ruler detected for {img_path}")
    elif best_match is None:
        logger.warning(f"No match found for {img_path}")

    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f)
