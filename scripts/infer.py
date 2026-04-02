# 추론 + 결과 이미지 저장
import os
import cv2

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner

def main():
    tok = CharTokenizer()
    model = YoloIconCaptioner(
        yolo_weights="checkpoints/yolo/best.pt",
        tokenizer=tok,
        imgsz=640,
        feature_layer_idx=21,
        roi_out=3,
        max_len=24,
        conf=0.25,
        iou=0.45,
        topk=10,
        decoder_cfg=dict(d_model=256, nhead=4, num_layers=3, ffn=1024),
    ).to("cuda")
    model.eval()

    img_path = "test.jpg"
    out = model.forward_infer(img_path, device="cuda")
    boxes = out["boxes_xyxy"]
    texts = out["texts"]

    img = cv2.imread(img_path)
    for (x1,y1,x2,y2), t in zip(boxes, texts):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, t, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/infer_vis.jpg", img)
    print("Saved outputs/infer_vis.jpg")
    print("texts:", texts)

if __name__ == "__main__":
    main()