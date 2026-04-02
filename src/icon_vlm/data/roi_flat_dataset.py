from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from icon_vlm.data.datasets import IconCaptionDataset  # 기존 jsonl 파싱/전처리 로직이 있으면 재사용
from icon_vlm.models.tokenizer import CharTokenizer
from icon_vlm.models.preprocess import letterbox, xyxy_orig_to_letterbox


def _norm_label(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


@dataclass
class RoiSample:
    # 한 ROI(=한 라벨) 샘플
    image_path: str
    orig_hw: Tuple[int, int]
    # 원본 좌표 (jsonl이 어떤 좌표를 담는지에 따라 조정 가능)
    box_xyxy_orig: List[float]
    text: str


class ROIFlatDataset(Dataset):
    """
    jsonl 레코드(이미지 1장에 ROI 여러개)를
    (이미지, 단일 ROI) 단위 샘플로 펼친 Dataset

    __getitem__은:
      - image를 letterbox + normalize 해서 [3,imgsz,imgsz]
      - 해당 ROI box를 letterbox 좌표계로 변환해서 [4]
      - text를 token ids [max_len]
    를 반환.

    return:
      image: FloatTensor [3,imgsz,imgsz]
      box_lb: FloatTensor [4]  (xyxy in letterbox coords)
      text_ids: LongTensor [max_len]
      label_str: str (디버그/분포 확인용)
    """
    def __init__(self, manifest_jsonl: str, tokenizer: CharTokenizer, imgsz: int = 640, max_len: int = 24):
        assert os.path.isfile(manifest_jsonl), f"missing manifest: {manifest_jsonl}"
        self.manifest = manifest_jsonl
        self.tok = tokenizer
        self.imgsz = int(imgsz)
        self.max_len = int(max_len)

        # jsonl 전체를 한번 펼쳐서 index 만들기 (데이터가 매우 크면 streaming으로 바꿔야 함)
        self.samples: List[RoiSample] = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                path = rec.get("image_path") or rec.get("path") or rec.get("img_path")
                assert path, f"missing image_path at line {line_no}"
                if not os.path.isfile(path):
                    # 기존 코드가 base dir 붙이는 로직이 있다면 여기에서 맞춰줘야 함
                    raise FileNotFoundError(path)

                boxes = rec.get("boxes_xyxy") or rec.get("boxes")
                texts = rec.get("texts")

                if not isinstance(boxes, list) or not isinstance(texts, list) or len(boxes) != len(texts):
                    continue

                # 이미지 크기는 로딩해서 얻는다 (한번만)
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(path)
                h0, w0 = img.shape[:2]

                for b, t in zip(boxes, texts):
                    if b is None or len(b) != 4:
                        continue
                    lab = _norm_label(t)
                    if not lab:
                        continue
                    self.samples.append(
                        RoiSample(
                            image_path=path,
                            orig_hw=(h0, w0),
                            box_xyxy_orig=[float(x) for x in b],
                            text=lab,
                        )
                    )

        assert len(self.samples) > 0, "ROIFlatDataset has 0 samples. check manifest format."

        # label list for weighting
        self.labels = [_norm_label(s.text) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_preprocess(self, path: str):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(path)

        h0, w0 = img_bgr.shape[:2]
        img_lb, ratio, pad = letterbox(img_bgr, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        return x, (h0, w0), ratio, pad

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        x_np, (h0, w0), ratio, pad = self._load_and_preprocess(s.image_path)
        x = torch.from_numpy(x_np)  # [3,H,W]

        # orig -> letterbox coords
        box_orig = np.array(s.box_xyxy_orig, dtype=np.float32)[None, :]  # [1,4]
        box_lb = xyxy_orig_to_letterbox(box_orig, ratio, pad)[0]         # [4]
        box_lb = torch.tensor(box_lb, dtype=torch.float32)

        # text -> ids (BOS..EOS..PAD) 형태는 tokenizer 구현에 맞춤
        text_ids = self.tok.encode(s.text, max_len=self.max_len)  # 프로젝트 tokenizer에 encode가 있어야 함
        # 만약 encode가 없고 기존 파이프라인이 다른 함수면 여기만 바꿔주기
        text_ids = torch.tensor(text_ids, dtype=torch.long)

        return x, box_lb, text_ids, s.text