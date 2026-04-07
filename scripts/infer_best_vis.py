# More complete inference/visualization utility.
# Loads a best checkpoint, benchmarks latency, runs inference over images, and saves annotated outputs.

import os
import time
import glob
import argparse
import torch
import cv2

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.yolo_captioner import YoloIconCaptioner


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_best(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    print("YOLO weights from ckpt cfg:", cfg["paths"]["yolo_weights"])

    tok = CharTokenizer()
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

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def draw_boxes_and_text(img_bgr, boxes_xyxy, texts):
    """
    boxes_xyxy: numpy (N,4) in original image coords
    texts: list[str]
    """
    out = img_bgr.copy()
    for i, (b, t) in enumerate(zip(boxes_xyxy, texts)):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(out.shape[1] - 1, x2); y2 = min(out.shape[0] - 1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = t
        # label background
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - 6)
        cv2.rectangle(out, (x1, y_text - th - base), (x1 + tw + 6, y_text + base), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/captioner/best.pt")
    ap.add_argument("--image", type=str, default=None, help="single image path")
    ap.add_argument("--dir", type=str, default=None, help="directory of images")
    ap.add_argument("--outdir", type=str, default="outputs/infer_vis")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    assert os.path.exists(args.ckpt), f"ckpt not found: {args.ckpt}"
    model = load_best(args.ckpt, device)

    # 입력 이미지 목록
    paths = []
    if args.image:
        paths = [args.image]
    elif args.dir:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for e in exts:
            paths += glob.glob(os.path.join(args.dir, e))
        paths = sorted(paths)
    else:
        raise SystemExit("Provide --image or --dir")
    assert len(paths) > 0, "No images found."

    os.makedirs(args.outdir, exist_ok=True)

    # ---- 속도 측정 (첫 이미지로) ----
    bench_path = paths[0]
    print(f"\nBenchmark image: {bench_path}")

    def run_once():
        return model.forward_infer(bench_path, device=device)

    for _ in range(args.warmup):
        _ = run_once()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = run_once()
    sync(device)
    t1 = time.perf_counter()

    ms = (t1 - t0) / args.iters * 1000.0
    print(f"Avg latency: {ms:.2f} ms / image  (warmup={args.warmup}, iters={args.iters})")

    # ---- 결과 저장 ----
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print("skip (cannot read):", p)
            continue

        out = model.forward_infer(p, device=device)
        boxes = out["boxes_xyxy"]
        texts = out["texts"]

        vis = draw_boxes_and_text(img_bgr, boxes, texts)

        base = os.path.splitext(os.path.basename(p))[0]
        save_path = os.path.join(args.outdir, f"{base}_pred.jpg")
        cv2.imwrite(save_path, vis)

        print(f"saved: {save_path}  (N={len(texts)})")


if __name__ == "__main__":
    main()