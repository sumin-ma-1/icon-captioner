# sanity check 1
from ultralytics import YOLO

model = YOLO("checkpoints/yolo/best.pt")
for i, m in enumerate(model.model.model):
    print(i, m.__class__.__name__)