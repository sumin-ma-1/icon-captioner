# sanity check 1
# Very small sanity script that prints the layer/module names of the YOLO model.

from ultralytics import YOLO

model = YOLO("checkpoints/yolo/best.pt")
for i, m in enumerate(model.model.model):
    print(i, m.__class__.__name__)