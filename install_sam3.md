### SAM3

SAM3 is not installed by `uv sync`. From this directory, clone the repo:

```bash
git clone https://github.com/facebookresearch/sam3.git
```

Download `sam3.1_multiplex.pt` from the [SAM 3.1 Hugging Face repo](https://huggingface.co/facebook/sam3.1) and put it in `models/`. So the full path will be: `models/sam3.1_multiplex.pt`

`sam3_inference.py` reads the code from `sam3/` and the weights from `models/sam3.1_multiplex.pt`.
