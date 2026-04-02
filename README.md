## Setup
pip install -r requirements.txt

## Put YOLO weights
checkpoints/yolo/best.pt

## Prepare data
data/processed/train.jsonl
data/processed/val.jsonl

Each line:
{"image_path":"...","boxes_xyxy":[[x1,y1,x2,y2],...],"texts":["settings","wifi",...]}

## Sanity checks
python -m src.scripts.print_layers
python -m src.scripts.debug_preprocess_roi
python -m src.scripts.overfit_one_batch

## Train
python -m src.scripts.train

## Inference
python -m src.scripts.infer