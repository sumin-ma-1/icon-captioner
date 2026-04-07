# Visualizes predicted boxes and texts on a single image.
# Uses forward_infer and writes the annotated image to disk.

import os
import cv2
import numpy as np

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner


def draw_boxes(img_bgr, boxes, texts=None, color=(0, 255, 0)):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if texts is not None:
            t = str(texts[i])[:30]
            cv2.putText(out, t, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


def main():
    weights = os.environ["YOLO_WEIGHTS"]
    img_path = os.environ["IMG"]
    out_path = os.environ.get("OUT", "vis_boxes.jpg")

    tokenizer = CharTokenizer()
    model = YoloIconCaptioner(
        yolo_weights=weights,
        tokenizer=tokenizer,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        topk=10,
    )

    out = model.forward_infer(img_path, device="cpu")

    img = cv2.imread(img_path)
    assert img is not None, img_path

    vis = draw_boxes(img, out["boxes_xyxy"], out["texts"])
    cv2.imwrite(out_path, vis)
    print("saved:", out_path)
    print("image size:", img.shape[:2], "(H,W)")
    print("boxes max:", np.max(out["boxes_xyxy"]) if len(out["boxes_xyxy"]) else None)


if __name__ == "__main__":
    main()