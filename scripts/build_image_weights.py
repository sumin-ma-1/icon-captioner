# Builds per-image sampling weights from a training manifest.
# Uses inverse label frequency averaged over ROI labels and writes JSON weights.

import os, json, argparse, math
from collections import Counter

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
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="inverse frequency exponent. 1.0=strong, 0.5=moderate")
    ap.add_argument("--max_w", type=float, default=10.0,
                    help="cap for extreme rare labels")
    args = ap.parse_args()

    assert os.path.isfile(args.manifest), f"missing manifest: {args.manifest}"

    # 1) label freq
    freq = Counter()
    records = []
    for rec in iter_jsonl(args.manifest):
        texts = rec.get("texts", [])
        texts = [norm_label(t) for t in texts if norm_label(t)]
        records.append(texts)
        for t in texts:
            freq[t] += 1

    # 2) per-image weight
    # idea: 이미지에 있는 ROI들의 inverse-freq를 평균내서 weight로 씀
    weights = []
    for texts in records:
        if not texts:
            weights.append(1.0)
            continue

        invs = []
        for t in texts:
            f = freq.get(t, 1)
            inv = (1.0 / float(f)) ** args.alpha
            invs.append(inv)

        w = sum(invs) / len(invs)          # 평균
        w = min(w, args.max_w)             # cap
        weights.append(w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": args.manifest,
                "alpha": args.alpha,
                "max_w": args.max_w,
                "num_images": len(weights),
                "weights": weights,
                "top_labels": freq.most_common(50),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("saved:", args.out)
    print("num_images:", len(weights))
    print("weight stats: min={:.4f} mean={:.4f} max={:.4f}".format(min(weights), sum(weights)/len(weights), max(weights)))
    print("top5 labels:", freq.most_common(5))

if __name__ == "__main__":
    main()