import json
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.preprocess import letterbox, xyxy_orig_to_letterbox
from ..models.tokenizer import CharTokenizer

class IconCaptionDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer: CharTokenizer, imgsz: int = 640, max_len: int = 24):
        self.items = []
        self.tokenizer = tokenizer
        self.imgsz = imgsz
        self.max_len = max_len

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        path = it["image_path"]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(path)

        orig = img_bgr
        img_lb, ratio, pad = letterbox(orig, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        boxes_orig = np.array(it["boxes_xyxy"], dtype=np.float32)  # [N,4] original coords
        boxes_lb = xyxy_orig_to_letterbox(boxes_orig, ratio, pad).astype(np.float32)

        texts = it["texts"]
        text_ids = [self.tokenizer.encode(s, self.max_len) for s in texts]  # [N, T]

        return {
            "image": torch.from_numpy(img),                      # [3,imgsz,imgsz]
            "boxes_lb": torch.from_numpy(boxes_lb),              # [N,4] letterbox coords
            "text_ids": torch.tensor(text_ids, dtype=torch.long) # [N,T]
        }