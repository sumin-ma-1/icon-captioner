# sanity check 2: ROI가 맞는지 시각 확인
# Sanity-checks ROI preprocessing and letterbox coordinate conversion.
# Loads one sample, visualizes letterboxed GT boxes, and checks ROI token shapes through the model.

"""
manifest 샘플 1개를 읽고

letterbox 후 박스를 그림으로 저장

ROI 토큰 shape 출력

decoder 입력 전까지가 정상인지 확인하는 디버그 스크립트
"""
import json
import cv2
import numpy as np
import torch

from icon_vlm.models.preprocess import letterbox, xyxy_orig_to_letterbox
from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner
from icon_vlm.models.roi import roi_tokens_roi_align

def main():
    tok = CharTokenizer()
    model = YoloIconCaptioner(
        yolo_weights="checkpoints/yolo/best.pt",
        tokenizer=tok,
        imgsz=640,
        feature_layer_idx=21,
        roi_out=3,
        max_len=24,
        decoder_cfg=dict(d_model=256, nhead=4, num_layers=3, ffn=1024),
    ).to("cuda")
    model.freeze_yolo()

    # manifest 1줄 읽기
    with open("data/processed/train.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(next(f))

    img_bgr = cv2.imread(sample["image_path"])
    boxes_orig = np.array(sample["boxes_xyxy"], dtype=np.float32)

    img_lb, ratio, pad = letterbox(img_bgr, (640, 640))
    boxes_lb = xyxy_orig_to_letterbox(boxes_orig, ratio, pad)

    # 시각화: letterbox 이미지에 letterbox 박스 그리기
    vis = img_lb.copy()
    for b in boxes_lb:
        x1,y1,x2,y2 = b.astype(int)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite("debug_letterbox_boxes.jpg", vis)
    print("Saved debug_letterbox_boxes.jpg")

    # 모델 feature/ROI 토큰 shape 확인 (GT 박스 기준)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = (img_rgb.astype(np.float32)/255.0).transpose(2,0,1)
    x = torch.from_numpy(x).unsqueeze(0).to("cuda")

    pred, feat = model._capture_feature(x)
    print("feat shape:", tuple(feat.shape))  # [1,C,Hf,Wf]

    boxes_lb_t = torch.from_numpy(boxes_lb).to("cuda")
    tokens = roi_tokens_roi_align(feat, boxes_lb_t, imgsz=640, roi_out=3)
    print("roi tokens:", tuple(tokens.shape)) # [N,9,C]

if __name__ == "__main__":
    main()