from typing import Tuple
import torch
from torchvision.ops import roi_align

def roi_tokens_roi_align(
    feat: torch.Tensor,
    boxes_xyxy_lb: torch.Tensor,
    imgsz: int,
    roi_out: int = 3,
):
    """
    feat: [B,C,Hf,Wf] (letterbox 이미지 기준 feature)
    boxes_xyxy_lb: [N,4] (letterbox 좌표계, pixel)
    imgsz: letterbox 입력 크기 (예: 640)
    return tokens: [N, roi_out*roi_out, C]
    """
    if boxes_xyxy_lb.numel() == 0:
        return torch.zeros((0, roi_out*roi_out, feat.size(1)), device=feat.device)

    if feat.size(0) != 1:
        raise ValueError("roi_tokens_roi_align는 B=1에서 쓰는 헬퍼입니다. 배치 확장은 yolo_captioner에서 처리합니다.")

    rois = torch.cat([
        torch.zeros((boxes_xyxy_lb.size(0), 1), device=boxes_xyxy_lb.device, dtype=boxes_xyxy_lb.dtype),
        boxes_xyxy_lb
    ], dim=1)  # [N,5]

    Hf = feat.size(-2)
    spatial_scale = Hf / float(imgsz)   # letterbox 기준이므로 imgsz 사용

    roi_feat = roi_align(
        input=feat,
        boxes=rois,
        output_size=roi_out,
        spatial_scale=spatial_scale,
        sampling_ratio=2,
        aligned=True
    )  # [N,C,roi_out,roi_out]

    tokens = roi_feat.permute(0, 2, 3, 1).contiguous().view(roi_feat.size(0), roi_out*roi_out, roi_feat.size(1))
    return tokens