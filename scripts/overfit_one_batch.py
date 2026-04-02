# sanity check 3: 학습 됨을 확인
"""
한 배치만 계속 학습해서 loss가 내려가면

ROI가 맞고

디코더 학습 경로가 정상이고

텍스트 토큰도 정상이라는 뜻
"""
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner
from icon_vlm.data.datasets import IconCaptionDataset
from icon_vlm.data.collate import collate_icon_caption

def main():
    cfg = yaml.safe_load(open("configs/train.yaml", "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = CharTokenizer()
    ds = IconCaptionDataset(cfg["paths"]["train_manifest"], tok, imgsz=cfg["data"]["imgsz"], max_len=cfg["data"]["max_len"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_icon_caption)

    model = YoloIconCaptioner(
        yolo_weights=cfg["paths"]["yolo_weights"],
        tokenizer=tok,
        imgsz=cfg["data"]["imgsz"],
        feature_layer_idx=cfg["model"]["feature_layer_idx"],
        roi_out=cfg["model"]["roi_out"],
        max_len=cfg["model"]["max_len"],
        conf=cfg["model"]["conf"],
        iou=cfg["model"]["iou"],
        topk=cfg["model"]["topk"],
        decoder_cfg=cfg["model"]["decoder"]
    ).to(device)

    if cfg["train"]["freeze_yolo"]:
        model.freeze_yolo()

    # config가 또 문자열로 들어올 수 있으니까 scripts/overfit_one_batch.py에서 optimizer 만들기 직전에 강제 변환
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd
    )

    images, gt_boxes_list, gt_text_ids = next(iter(dl))
    images = images.to(device)
    gt_text_ids = gt_text_ids.to(device)

    for step in range(100):
        model.train()
        out = model.forward_train(images, gt_boxes_list, gt_text_ids)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # 10 step 마다 예측 문자열 찍기: 실제로 텍스트를 맞추는지 확인
        if (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                dbg = model.forward_train_decode(images, gt_boxes_list)
                pred_texts = dbg["pred_texts"]

                # GT 텍스트도 사람이 읽게 복원 (R개)
                gt_texts = [tok.decode(row.tolist()) for row in gt_text_ids]

                print("---- sample compare (first 5 rois) ----")
                for i in range(min(5, len(pred_texts), len(gt_texts))):
                    print(f"[{i}] GT : {gt_texts[i]}")
                    print(f"    PR : {pred_texts[i]}")

        if (step + 1) % 10 == 0:
            print(f"step {step+1:03d} loss={loss.item():.4f}")

if __name__ == "__main__":
    main()