# Reads a JSONL manifest and reports label distribution statistics.
# Prints top labels, word-count distribution, and top words from ROI texts.

import os
import json
from collections import Counter, defaultdict

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] JSON decode failed at line {line_no}: {path}")
                continue

def normalize_label(s: str) -> str:
    # 학습에서 쓰는 min_filter랑 최대한 비슷하게 맞추는 게 좋지만
    # 여기선 분포만 보려는 거라 간단 정규화만 진행
    if s is None:
        return ""
    s = s.strip().lower()
    # 연속 공백 정리
    s = " ".join(s.split())
    return s

def main():
    # 기본 경로: configs/train.yaml의 paths.train_manifest를 쓰면 제일 좋지만
    # 지금은 빠르게 하려고 env/기본값으로 처리
    manifest = os.environ.get("TRAIN_JSONL", "data/processed/icon_caption_jsonl_gt/train.jsonl")
    topk = int(os.environ.get("TOPK", "50"))

    assert os.path.isfile(manifest), f"missing TRAIN_JSONL: {manifest}"

    cnt = Counter()
    n_images = 0
    n_rois = 0
    empty = 0
    by_len = Counter()
    multiword = 0

    # 추가: 단어별 빈도도 같이 보고 싶으면 여기서 집계
    word_cnt = Counter()

    for rec in iter_jsonl(manifest):
        n_images += 1
        texts = rec.get("texts", [])
        if not isinstance(texts, list):
            continue

        for t in texts:
            n_rois += 1
            lab = normalize_label(t)
            if not lab:
                empty += 1
                continue
            cnt[lab] += 1

            # 길이/단어수 통계
            words = lab.split()
            by_len[len(words)] += 1
            if len(words) >= 2:
                multiword += 1

            for w in words:
                word_cnt[w] += 1

    print("=== Label Distribution (train) ===")
    print(f"manifest: {manifest}")
    print(f"num images: {n_images}")
    print(f"num rois:   {n_rois}")
    print(f"empty text: {empty} ({(empty/max(1,n_rois))*100:.2f}%)")
    print()

    # Top labels
    print(f"--- Top {topk} labels ---")
    total_nonempty = sum(cnt.values())
    for i, (lab, c) in enumerate(cnt.most_common(topk), 1):
        frac = (c / max(1, total_nonempty)) * 100.0
        print(f"{i:02d}. {lab:25s}  {c:6d}  ({frac:6.2f}%)")

    print()
    print("--- Word-count distribution (label length in words) ---")
    for k in sorted(by_len.keys()):
        c = by_len[k]
        print(f"{k} words: {c:6d}  ({(c/max(1,total_nonempty))*100:.2f}%)")
    print(f"multiword labels: {multiword} ({(multiword/max(1,total_nonempty))*100:.2f}%)")
    print()

    print(f"--- Top {topk} words (from labels) ---")
    for i, (w, c) in enumerate(word_cnt.most_common(topk), 1):
        print(f"{i:02d}. {w:20s} {c:6d}")

if __name__ == "__main__":
    main()