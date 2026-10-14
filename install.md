### Installation

Install uv: https://docs.astral.sh/uv/getting-started/installation/

From this directory, the default environment is PyTorch (CUDA 13), flash-attn, DINOv3, LoMa, PaddleOCR, and RF-DETR:

```bash
uv sync
```

That install also includes the notebook stack used for leaf-area plots: Jupyter, pandas, pyarrow, seaborn, and Plotly.

LightGlue and CLIP are optional, and can be installed separately or together:

```bash
uv sync --extra lightglue
uv sync --extra clip
uv sync --extra lightglue --extra clip
```

SAM3 is separate. See [install_sam3.md](install_sam3.md).

### Models

Download the DINO and RF-DETR weights from URL and put them in `models/`.

- DINOv3 ViT-L/16, used at `models/dinov3-vitl16-pretrain-lvd1689m`. Original weights: [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
- Finetuned RF-DETR Seg Nano ruler checkpoint, used at `models/rf_detr_seg_nano_rulers.pth`. Original model: [Roboflow/rf-detr-seg-nano](https://huggingface.co/Roboflow/rf-detr-seg-nano).

CLIP is optional. Download it from URL2 and put it in `models/PE-Core-L-14-336/` (`open_clip_pytorch_model.bin`). Original weights: [facebook/PE-Core-L14-336](https://huggingface.co/facebook/PE-Core-L14-336).
