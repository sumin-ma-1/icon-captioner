import os
import cv2
import numpy as np
import torch

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner

def main():
    weights = os.environ["YOLO_WEIGHTS"]
    img_path = os.environ["IMG"]

    device = "cpu"  # 우선 CPU로
    tok = CharTokenizer()  # 구현에 맞게 init 필요하면 맞춰주기

    m = YoloIconCaptioner(
        yolo_weights=weights,
        tokenizer=tok,
        imgsz=640,
        conf=float(os.environ.get("CONF", "0.25")),
        iou=float(os.environ.get("IOU", "0.45")),
        topk=int(os.environ.get("TOPK", "10")),
    ).to(device)

    out = m.forward_infer(img_path, device=device)
    print("num boxes:", len(out["texts"]))
    print("texts:", out["texts"][:5])
    print("boxes:", out["boxes_xyxy"][:5])

    # crop 체크 (orig coords 기준)
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    for i, b in enumerate(out["boxes_xyxy"][:min(5, len(out["boxes_xyxy"]))]):
        x1,y1,x2,y2 = map(int, b)
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        crop = img[y1:y2, x1:x2]
        print(f"[crop {i}] shape:", crop.shape)

if __name__ == "__main__":
    main()