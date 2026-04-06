# Builds per-ROI sampling weights from a training manifest.
# Computes inverse-frequency weights for each ROI label and saves them in JSON.

import os, json, argparse
from collections import Counter

def norm_label(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/icon_caption_jsonl_gt/train.jsonl")
    ap.add_argument("--out", default="data/processed/icon_caption_jsonl_gt/train_roi_weights.json")
    ap.add_argument("--alpha", type=float, default=1.0, help="inv-freq exponent (1.0~2.0)")
    ap.add_argument("--smooth", type=float, default=1.0, help="additive smoothing on freq")
    ap.add_argument("--max_w", type=float, default=50.0)
    args = ap.parse_args()

    assert os.path.isfile(args.manifest), f"missing: {args.manifest}"

    # 1) ROI labels list (flatten)
    roi_labels = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            texts = rec.get("texts", [])
            for t in texts:
                lab = norm_label(t)
                if lab:
                    roi_labels.append(lab)

    assert len(roi_labels) > 0, "no roi labels found"

    # 2) freq
    freq = Counter(roi_labels)

    # 3) weight per ROI
    weights = []
    for lab in roi_labels:
        f = float(freq[lab]) + float(args.smooth)
        w = (1.0 / f) ** float(args.alpha)
        if w > args.max_w:
            w = args.max_w
        weights.append(w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": args.manifest,
                "alpha": args.alpha,
                "smooth": args.smooth,
                "max_w": args.max_w,
                "num_rois": len(weights),
                "weights": weights,
                "top_labels": freq.most_common(50),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("saved:", args.out)
    print("num_rois:", len(weights))
    print("top5:", freq.most_common(5))

if __name__ == "__main__":
    main()