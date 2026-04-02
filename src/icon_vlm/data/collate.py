from typing import List, Dict, Any, Tuple
import torch

def collate_icon_caption(batch: List[Dict[str, Any]]):
    images = torch.stack([b["image"] for b in batch], dim=0)  # [B,3,H,W]

    gt_boxes_list = [b["boxes_lb"] for b in batch]            # list of [Ni,4]
    # text_ids는 [Ni,T] -> batch flatten [R,T]
    text_ids = torch.cat([b["text_ids"] for b in batch], dim=0)

    return images, gt_boxes_list, text_ids