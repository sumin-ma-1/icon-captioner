import os
import json
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner
from icon_vlm.training.train_loop import train_one_epoch, eval_one_epoch

from icon_vlm.data.roi_flat_dataset import ROIFlatDataset
from icon_vlm.data.collate_roi_flat import collate_roi_flat

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # device 선택: cuda > mps > cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    # ---- tokenizer / dataset ----
    tok = CharTokenizer()

    train_manifest = cfg["paths"]["train_manifest"]
    val_manifest = cfg["paths"].get("val_manifest", None)

    train_manifest = cfg["paths"]["train_manifest"]
    val_manifest = cfg["paths"].get("val_manifest", None)
    if val_manifest is None:
        val_manifest = train_manifest

    # ---- ROI-flat datasets ----
    ds_train = ROIFlatDataset(
        train_manifest, tok,
        imgsz=cfg["data"]["imgsz"],
        max_len=cfg["data"]["max_len"],
    )

    # ROI-level weights (필수로 쓰는 걸 추천)
    roi_weights_path = cfg["paths"].get("train_roi_weights", None)

    if roi_weights_path:
        with open(roi_weights_path, "r", encoding="utf-8") as f:
            wobj = json.load(f)
        weights = torch.tensor(wobj["weights"], dtype=torch.double)
        assert len(weights) == len(ds_train), f"roi weights len {len(weights)} != dataset len {len(ds_train)}"

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),  # 1 epoch에 뽑을 ROI 샘플 수
            replacement=True
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=int(cfg["train"]["batch_size"]),
            sampler=sampler,
            shuffle=False,  # sampler 쓰면 shuffle 금지
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collate_roi_flat,
            pin_memory=bool(cfg["train"].get("pin_memory", False)),
        )

        # ---- sanity check (run once) ----
        print("num train ROIs:", len(ds_train))

        b = next(iter(dl_train))
        images, gt_boxes_list, gt_text_ids = b

        print("images.shape:", images.shape)
        print("len(gt_boxes_list):", len(gt_boxes_list))
        print("gt_boxes_list[0].shape:", gt_boxes_list[0].shape)
        print("gt_text_ids.shape:", gt_text_ids.shape)
        print("--------------------------------")

    else:
        # sampler 없이도 돌릴 수는 있지만, 분포 붕괴 막는 목적이면 roi_weights 쓰는 게 핵심
        dl_train = DataLoader(
            ds_train,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            collate_fn=collate_roi_flat,
            pin_memory=bool(cfg["train"].get("pin_memory", False)),
        )

    # val_manifest을 안 넣었을 때만 fallback
    if val_manifest is None:
        val_manifest = train_manifest

    ds_val = ROIFlatDataset(
        val_manifest, tok,
        imgsz=cfg["data"]["imgsz"],
        max_len=cfg["data"]["max_len"],
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,                  # val은 절대 sampler/밸런싱 하지 말기
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_roi_flat,
        pin_memory=bool(cfg["train"].get("pin_memory", False)),
    )

    # ---- model ----
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

    # ---- optimizer ----
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
    )

    # ---- train control ----
    epochs = int(cfg["train"].get("epochs", 10))
    amp = bool(cfg["train"].get("amp", True))
    log_every = int(cfg["train"].get("log_every", 10))

    patience = int(cfg["train"].get("patience", 5))  # early stopping patience
    best_val = float("inf")
    bad_epochs = 0

    ckpt_dir = cfg["paths"].get("ckpt_dir", "checkpoints/captioner")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        tr = train_one_epoch(model, dl_train, opt, device=device, amp=amp, log_every=log_every)
        print(f"[epoch {epoch}] train loss={tr['loss']:.4f} tok_acc={tr['token_acc']:.3f}")

        va = eval_one_epoch(
            model, dl_val,
            device=device,
            amp=amp,
            log_every=log_every,
            log_samples=8,     # 추가
            sample_step=0,     # 추가 (첫 배치에서 찍기)
        )
        print(f"[epoch {epoch}] val   loss={va['loss']:.4f} tok_acc={va['token_acc']:.3f}")

        # epoch ckpt 저장
        epoch_path = os.path.join(ckpt_dir, f"captioner_epoch{epoch}.pt")
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "train": tr, "val": va, "cfg": cfg},
            epoch_path
        )

        # 이전 파일 삭제
        if epoch >= 3:
            old_path = os.path.join(ckpt_dir, f"captioner_epoch{epoch-3}.pt")
            if os.path.exists(old_path):
                os.remove(old_path)

        # best 저장 + early stopping
        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            bad_epochs = 0
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "train": tr, "val": va, "cfg": cfg},
                best_path
            )
            print(f"Best updated: epoch={epoch} best_val={best_val:.4f} -> {best_path}")
        else:
            bad_epochs += 1
            print(f"no val improvement. bad_epochs={bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print(f"Early stop: best_val={best_val:.4f}")
                break


if __name__ == "__main__":
    main()