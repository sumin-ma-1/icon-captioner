import os, json, argparse
from collections import Counter
import numpy as np

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def norm_label(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/icon_caption_jsonl_gt/train.jsonl")
    ap.add_argument("--out", default="data/processed/icon_caption_jsonl_gt/train_image_weights.json")
    ap.add_argument("--alpha", type=float, default=1.0, help="inv-freq exponent. 1.5~2.0 추천")
    ap.add_argument("--mode", type=str, default="minfreq",
                    choices=["mean", "max", "minfreq", "topk_mean"],
                    help="image weight aggregation mode")
    ap.add_argument("--topk", type=int, default=2, help="mode=topk_mean일 때 top-k inv만 평균")
    ap.add_argument("--max_w", type=float, default=50.0, help="cap for extreme rare labels")
    ap.add_argument("--smooth", type=float, default=1.0, help="freq smoothing (additive)")
    args = ap.parse_args()

    assert os.path.isfile(args.manifest), f"missing manifest: {args.manifest}"

    # 1) collect per-image labels + global freq
    freq = Counter()
    img_labels = []
    for rec in iter_jsonl(args.manifest):
        texts = rec.get("texts", [])
        labs = [norm_label(t) for t in texts]
        labs = [l for l in labs if l]
        img_labels.append(labs)
        for l in labs:
            freq[l] += 1

    # 2) compute image weights
    weights = []
    for labs in img_labels:
        if not labs:
            weights.append(1.0)
            continue

        invs = []
        for l in labs:
            f = float(freq[l]) + float(args.smooth)   # smoothing
            invs.append((1.0 / f) ** args.alpha)

        invs = np.array(invs, dtype=np.float64)

        if args.mode == "mean":
            w = float(invs.mean())
        elif args.mode == "max":
            w = float(invs.max())
        elif args.mode == "minfreq":
            # 가장 희소한 라벨(=freq 최소)을 기준으로 weight 부여
            min_f = min(freq[l] for l in labs)
            w = float((1.0 / (float(min_f) + float(args.smooth))) ** args.alpha)
        elif args.mode == "topk_mean":
            k = max(1, min(args.topk, len(invs)))
            top = np.sort(invs)[-k:]
            w = float(top.mean())
        else:
            raise ValueError(args.mode)

        w = min(w, args.max_w)
        weights.append(w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": args.manifest,
                "alpha": args.alpha,
                "mode": args.mode,
                "topk": args.topk,
                "max_w": args.max_w,
                "smooth": args.smooth,
                "num_images": len(weights),
                "weights": weights,
                "top_labels": freq.most_common(50),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    w = np.array(weights, dtype=np.float64)
    print("saved:", args.out)
    print("num_images:", len(weights))
    print(f"weight stats: min={w.min():.6f} mean={w.mean():.6f} max={w.max():.6f}")
    print("top5 labels:", freq.most_common(5))

if __name__ == "__main__":
    main()