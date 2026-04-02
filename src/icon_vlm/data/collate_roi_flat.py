from __future__ import annotations
from typing import List, Tuple
import torch


def collate_roi_flat(batch):
    """
    batch: list of (image [3,H,W], box_lb [4], text_ids [L], label_str)

    기존 model.forward_train(images, gt_boxes_lb_list, gt_text_ids) 형태를 맞추려면:
      - images: [B,3,H,W]
      - gt_boxes_lb_list: length B, each [Ni,4]
      - gt_text_ids: [R,L] where R=sum Ni

    ROI-flat에서는 각 샘플이 ROI 1개라서:
      - 각 이미지마다 ROI 1개로 두면 가장 단순함.
      - 즉 B = batch_size, Ni=1, R=B

    return:
      images: [B,3,H,W]
      gt_boxes_lb_list: list length B, each [1,4]
      gt_text_ids: [B,L]
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1].unsqueeze(0) for b in batch]  # each [1,4]
    text_ids = torch.stack([b[2] for b in batch], dim=0)
    return images, boxes, text_ids
