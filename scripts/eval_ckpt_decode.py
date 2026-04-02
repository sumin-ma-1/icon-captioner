import os, glob
import yaml
import torch
from torch.utils.data import DataLoader

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner

from icon_vlm.data.roi_flat_dataset import ROIFlatDataset
from icon_vlm.data.collate_roi_flat import collate_roi_flat


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(cfg, tok, device):
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
        decoder_cfg=cfg["model"]["decoder"],
    ).to(device)

    if cfg["train"].get("freeze_yolo", False):
        model.freeze_yolo()

    return model


def main():
    device = pick_device()
    print("device:", device)

    # 어떤 ckpt 볼지
    ckpts = sorted(glob.glob("checkpoints/captioner/captioner_epoch*.pt"))
    assert len(ckpts) > 0, "no ckpt found at checkpoints/captioner/captioner_epoch*.pt"

    pick = []
    best = "checkpoints/captioner/best.pt"
    if os.path.exists(best):
        pick.append(best)
    pick += [p for p in ckpts if any(p.endswith(f"epoch{i}.pt") for i in [20,21,22])]
    if not pick:
        pick = [ckpts[0], ckpts[-1]]

    print("picked ckpts:")
    for p in pick:
        print(" -", p)

    # 각 ckpt마다 cfg로 dataset/model을 맞춰서 평가
    for ckpt_path in pick:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["cfg"] if isinstance(ckpt, dict) and "cfg" in ckpt else yaml.safe_load(open("configs/train.yaml", "r", encoding="utf-8"))

        tok = CharTokenizer()

        val_manifest = cfg["paths"].get("val_manifest", None) or cfg["paths"]["train_manifest"]
        ds = ROIFlatDataset(val_manifest, tok, imgsz=cfg["data"]["imgsz"], max_len=cfg["data"]["max_len"])
        torch.manual_seed(0)
        dl = DataLoader(
            ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_roi_flat,
        )
        images, gt_boxes_list, gt_text_ids = next(iter(dl))

        # 샘플 배치 1개 고정
        images, gt_boxes_list, gt_text_ids = next(iter(dl))
        gt_texts = [tok.decode(row.tolist()) for row in gt_text_ids]
        images = images.to(device)

        model = load_model(cfg, tok, device)

        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()

        dbg = model.forward_train_decode(images, gt_boxes_list, device=device, max_len=cfg["data"]["max_len"])
        pred_texts = dbg["pred_texts"][:10]

        print("\n===", os.path.basename(ckpt_path), "===")
        for i, ptxt in enumerate(pred_texts):
            gtxt = gt_texts[i] if i < len(gt_texts) else "<no-gt>"
            print(f"{i:02d} GT={gtxt:15s} | PRED={ptxt}")


if __name__ == "__main__":
    main()