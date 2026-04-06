# Loads the train manifest and compares label distribution between random shuffle and weighted sampling.
# Uses a dataset loader and decoder to count GT labels from batches and print top-k distributions.

import os
import json
import yaml
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.data.datasets import IconCaptionDataset
from icon_vlm.data.collate import collate_icon_caption


def norm_label(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def load_weights(path: str, ds_len: int) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    w = torch.tensor(obj["weights"], dtype=torch.double)
    assert len(w) == ds_len, f"weights len {len(w)} != dataset len {ds_len}"
    return w


def count_labels_from_loader(loader, n_steps: int = 200) -> Counter:
    """
    loader에서 n_steps 만큼만 돌면서 GT label 분포 세기
    (GT는 collate에서 gt_text_ids도 나오지만 여기서는 dataset->jsonl의 texts 분포를 보고 싶은 거라
     loader의 batch로부터 text ids를 decode하는 대신, dataset 레벨에서 세면 더 정확함)
    """
    # 하지만 지금은 loader로부터 바로 counts를 만들기 위해
    # gt_text_ids를 tokenizer로 디코드하는 대신
    # collate가 리턴하는 gt_text_ids를 그대로 쓰면 text 복원이 애매
    #
    # 가장 간단/확실한 방법:
    #   loader가 뽑아준 샘플 인덱스를 알 수 없으니
    #   샘플당 texts는 loader에서 직접 못 꺼냄.
    #
    # 그래서 여기서는 batch의 gt_text_ids를 decode해서 세는 방식을 제공.
    # (tokenizer.decode가 이미 프로젝트에 있으니 이게 제일 현실적)
    raise NotImplementedError


def main():
    cfg = yaml.safe_load(open("configs/train.yaml", "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tok = CharTokenizer()

    ds = IconCaptionDataset(
        cfg["paths"]["train_manifest"],
        tok,
        imgsz=cfg["data"]["imgsz"],
        max_len=cfg["data"]["max_len"],
    )

    # -------------------------
    # 1) baseline loader (shuffle)
    # -------------------------
    dl_base = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_icon_caption,
    )

    # -------------------------
    # 2) balanced loader (sampler)
    # -------------------------
    wpath = cfg["paths"].get("train_image_weights", None)
    assert wpath is not None, "configs/train.yaml에 paths.train_image_weights를 넣어줘"
    assert os.path.isfile(wpath), f"missing weights file: {wpath}"

    weights = load_weights(wpath, len(ds))
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)

    dl_bal = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_icon_caption,
    )

    # -------------------------
    # 분포 측정: gt_text_ids 디코드해서 라벨 카운트
    # -------------------------
    def count_from_dl(dl, n_batches=200):
        cnt = Counter()
        total = 0
        for i, (images, gt_boxes_list, gt_text_ids) in enumerate(dl):
            # gt_text_ids: [R, L] (R=sum ROIs in batch)
            # 각 row를 decode해서 라벨로 집계
            for r in range(gt_text_ids.size(0)):
                lab = tok.decode(gt_text_ids[r].tolist())
                lab = norm_label(lab)
                if not lab:
                    continue
                cnt[lab] += 1
                total += 1
            if i + 1 >= n_batches:
                break
        return cnt, total

    n_batches = int(os.environ.get("NB", "200"))

    base_cnt, base_total = count_from_dl(dl_base, n_batches=n_batches)
    bal_cnt, bal_total = count_from_dl(dl_bal, n_batches=n_batches)

    topk = int(os.environ.get("TOPK", "20"))

    print(f"\n=== Using {n_batches} batches ===")
    print(f"[baseline] total rois counted: {base_total}")
    for i, (lab, c) in enumerate(base_cnt.most_common(topk), 1):
        print(f"{i:02d}. {lab:25s} {c:6d} ({c/max(1,base_total)*100:6.2f}%)")

    print(f"\n[balanced] total rois counted: {bal_total}")
    for i, (lab, c) in enumerate(bal_cnt.most_common(topk), 1):
        print(f"{i:02d}. {lab:25s} {c:6d} ({c/max(1,bal_total)*100:6.2f}%)")

    # 간단 지표: top1 비율이 얼마나 줄었는지
    if base_cnt:
        base_top1 = base_cnt.most_common(1)[0][1] / max(1, base_total)
    else:
        base_top1 = 0.0
    if bal_cnt:
        bal_top1 = bal_cnt.most_common(1)[0][1] / max(1, bal_total)
    else:
        bal_top1 = 0.0

    print("\n--- sanity ---")
    print(f"baseline top1 share: {base_top1*100:.2f}%")
    print(f"balanced top1 share: {bal_top1*100:.2f}%")
    print("기대: balanced 쪽 top1 비율이 확 내려가고, top-k 라벨 분포가 더 평평해져야 함.")

if __name__ == "__main__":
    main()