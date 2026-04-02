# 단일 PyTorch nn.Module / 추론도 predict() 없이 전처리→forward→NMS까지 내부에서 처리
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO

from .decoder import TinyTransformerDecoder
from .tokenizer import CharTokenizer
from .preprocess import letterbox, xyxy_orig_to_letterbox, xyxy_letterbox_to_orig
from .roi import roi_tokens_roi_align

# =========================
# decode + NMS (ultralytics 의존 제거)
# =========================
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        cur = boxes[i].unsqueeze(0)
        rest = boxes[idxs[1:]]
        ious = box_iou_xyxy(cur, rest).squeeze(0)
        idxs = idxs[1:][ious <= iou_thres]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(dim=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def decode_pred_to_det(pred: torch.Tensor) -> torch.Tensor:
    """
    확인한 출력: out[0] shape = (1, 5, 8400)
    - (B, no, hw), no=4+nc
    - 여기선 nc=1이라고 가정 가능 (5=4+1)

    return:
      det_all: [hw, 6] = x1 y1 x2 y2 conf cls (cls는 0)
      (좌표는 'letterbox 입력 스케일' 기준이라고 가정)
    """
    # out이 tuple이면 out[0]만
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    if pred.dim() == 3 and pred.size(0) == 1 and pred.size(1) >= 5:
        p = pred.permute(0, 2, 1)[0]   # [hw, no]
        xywh = p[:, :4]                # [hw,4]
        cls_probs = p[:, 4:]           # [hw,1] (nc=1)

        if cls_probs.size(1) == 1:
            conf = cls_probs[:, 0]
            cls = torch.zeros_like(conf)
        else:
            conf, cls = cls_probs.max(dim=1)

        xyxy = xywh_to_xyxy(xywh)
        det_all = torch.cat([xyxy, conf.unsqueeze(1), cls.float().unsqueeze(1)], dim=1)  # [hw,6]
        return det_all

    raise RuntimeError(f"Unsupported pred shape for decode: {tuple(pred.shape)}")

class YoloIconCaptioner(nn.Module):
    """
    단일 nn.Module:
      - YOLO backbone/neck forward 1회
      - layer21 feature hook으로 추출
      - Detect output -> NMS -> letterbox 좌표 bbox
      - ROIAlign -> tokens -> tiny decoder -> text
    """
    def __init__(
        self,
        yolo_weights: str,
        tokenizer: CharTokenizer,
        imgsz: int = 640,
        feature_layer_idx: int = 21,
        roi_out: int = 3,
        max_len: int = 24,
        conf: float = 0.25,
        iou: float = 0.45,
        topk: int = 10,
        decoder_cfg: Dict[str, Any] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.imgsz = imgsz
        self.feature_layer_idx = feature_layer_idx
        self.roi_out = roi_out
        self.max_len = max_len
        self.conf = conf
        self.iou = iou
        self.topk = topk

        y = YOLO(yolo_weights)
        self.yolo_core = y.model  # 내부 nn.Module
        self.yolo_core.eval()

        if decoder_cfg is None:
            decoder_cfg = dict(d_model=256, nhead=4, num_layers=3, ffn=1024)

        self.decoder = TinyTransformerDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=decoder_cfg["d_model"],
            nhead=decoder_cfg["nhead"],
            num_layers=decoder_cfg["num_layers"],
            ffn=decoder_cfg["ffn"],
            max_len=max_len
        )

    def freeze_yolo(self):
        for p in self.yolo_core.parameters():
            p.requires_grad = False

    def _capture_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,3,imgsz,imgsz] normalized float
        returns:
          pred: YOLO raw pred tensor (before NMS)
          feat: layer21 feature [B,C,Hf,Wf]
        """
        feats = {}

        def hook_fn(module, inp, out):
            feats["feat"] = out

        layer = self.yolo_core.model[self.feature_layer_idx]
        h = layer.register_forward_hook(hook_fn)
        try:
            out = self.yolo_core(x)
        finally:
            h.remove()

        # Ultralytics 모델 출력 타입은 버전에 따라 tuple/list일 수 있음
        if isinstance(out, (list, tuple)):
            pred = out[0]
        else:
            pred = out

        feat = feats["feat"]
        return pred, feat

    def _preprocess_np(self, img_bgr: np.ndarray):
        orig = img_bgr
        h0, w0 = orig.shape[:2]
        img_lb, ratio, pad = letterbox(orig, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img, (h0, w0), ratio, pad

    @torch.no_grad()
    def forward_infer(self, image_path: str, device: str = "cuda"):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)

        img, orig_hw, ratio, pad = self._preprocess_np(img_bgr)
        x = torch.from_numpy(img).unsqueeze(0).to(device)

        pred, feat = self._capture_feature(x)

        # 1) raw pred -> det_all [N,6] = x1 y1 x2 y2 conf cls  (letterbox coords)
        det_all = decode_pred_to_det(pred)  # [N,6] torch.Tensor

        # 2) conf filter
        keep = det_all[:, 4] >= float(self.conf)
        det_all = det_all[keep]
        if det_all.numel() == 0:
            return {"boxes_xyxy": np.zeros((0,4)), "texts": [], "orig_hw": orig_hw}

        # 3) NMS (pure torch)
        boxes_lb = det_all[:, :4]
        scores   = det_all[:, 4]
        keep_idx = nms_xyxy(boxes_lb, scores, iou_thres=float(self.iou))

        # 4) topk
        if keep_idx.numel() > self.topk:
            keep_idx = keep_idx[: self.topk]

        det = det_all[keep_idx]
        boxes_lb = det[:, :4]
        
        # ROI tokens from feat (B=1)
        tokens = roi_tokens_roi_align(feat, boxes_lb, imgsz=self.imgsz, roi_out=self.roi_out)

        pred_ids = self.decoder.forward_greedy(
            mem_tokens=tokens,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id
        )
        texts = [self.tokenizer.decode(row.tolist()) for row in pred_ids]

        # boxes back to original coords for reporting
        boxes_lb_np = boxes_lb.detach().cpu().numpy()
        boxes_orig = xyxy_letterbox_to_orig(boxes_lb_np, ratio, pad)

        return {"boxes_xyxy": boxes_orig, "texts": texts, "orig_hw": orig_hw}

    def forward_train(
        self,
        images: torch.Tensor,
        gt_boxes_lb_list: List[torch.Tensor],
        gt_text_ids: torch.Tensor,
    ):
        """
        images: [B,3,imgsz,imgsz] (letterbox + normalized)
        gt_boxes_lb_list: list length B, each [Ni,4] in letterbox coords
        gt_text_ids: [R, max_len] (R=sum Ni), already tokenized (BOS..EOS)
        """
        pred, feat = self._capture_feature(images)  # pred는 여기서 사용 안 해도 됨

        # GT rois로 ROI tokens 생성 (batch flatten)
        all_tokens = []
        for b in range(images.size(0)):
            boxes = gt_boxes_lb_list[b].to(images.device)
            tokens_b = roi_tokens_roi_align(feat[b:b+1], boxes, imgsz=self.imgsz, roi_out=self.roi_out)
            all_tokens.append(tokens_b)
        mem_tokens = torch.cat(all_tokens, dim=0)  # [R,Tm,C]

        input_ids = gt_text_ids[:, :-1]
        target_ids = gt_text_ids[:, 1:]

        logits, _loss_raw = self.decoder.forward_train(
            mem_tokens=mem_tokens,
            input_ids=input_ids,
            target_ids=target_ids,
            pad_id=self.tokenizer.pad_id
        )

        # label smoothing loss로 덮어쓰기
        ls = 0.1  # 보통 0.05~0.15. 먼저 0.1 시도
        pad_id = self.tokenizer.pad_id

        # logits: [R, L-1, V], target_ids: [R, L-1]
        V = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=ls,
        )

        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def forward_train_decode(
        self,
        images: torch.Tensor,
        gt_boxes_lb_list: List[torch.Tensor],
        device: Optional[str] = None,
        max_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        디버그용: 학습 배치에서 GT bbox로 ROIAlign을 하고
        decoder를 greedy decoding 해서 text를 뽑기

        이게 맞게 나오면:
          - 전처리(letterbox) / GT bbox 좌표계 / ROIAlign / tokenizer / decoder 경로가 정상

        Args:
            images: [B,3,imgsz,imgsz] letterbox+normalized
            gt_boxes_lb_list: length B, each [Ni,4] letterbox xyxy
            device: None이면 images.device 사용
            max_len: None이면 self.max_len 사용

        Returns:
            dict:
              - pred_texts: List[str] 길이 R (R=sum Ni)
              - pred_ids:  Tensor [R, L]
              - num_rois_per_image: List[int] length B
        """
        self.eval()
        if device is None:
            device = images.device
        if max_len is None:
            max_len = self.max_len

        images = images.to(device)

        # 1) YOLO forward로 feature map만 얻음 (pred는 디버그에선 불필요)
        _, feat = self._capture_feature(images)

        # 2) GT ROIs로 ROIAlign -> mem_tokens 생성 (batch flatten)
        all_tokens = []
        num_rois_per_image = []
        for b in range(images.size(0)):
            boxes = gt_boxes_lb_list[b].to(device)
            num_rois_per_image.append(int(boxes.size(0)))
            if boxes.numel() == 0:
                continue
            tokens_b = roi_tokens_roi_align(
                feat[b:b+1], boxes, imgsz=self.imgsz, roi_out=self.roi_out
            )  # [Ni, Tm, C]
            all_tokens.append(tokens_b)

        if len(all_tokens) == 0:
            return {"pred_texts": [], "pred_ids": torch.empty((0, 0), dtype=torch.long), "num_rois_per_image": num_rois_per_image}

        mem_tokens = torch.cat(all_tokens, dim=0)  # [R, Tm, C]

        # 3) Greedy decode
        # - forward_greedy 내부가 max_len을 self.decoder/max_len로 쓰는 구조면 별도 인자 불필요
        # - 만약 forward_greedy가 max_len 인자를 받는다면 아래처럼 넘겨주기
        try:
            pred_ids = self.decoder.forward_greedy(
                mem_tokens=mem_tokens,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                max_len=max_len,   # forward_greedy가 max_len 받는 경우만 동작
            )
        except TypeError:
            # forward_greedy가 max_len을 인자로 안 받는 구현이면 이 경로로
            pred_ids = self.decoder.forward_greedy(
                mem_tokens=mem_tokens,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
            )

        pred_texts = [self.tokenizer.decode(row.tolist()) for row in pred_ids]
        return {
            "pred_texts": pred_texts,
            "pred_ids": pred_ids,
            "num_rois_per_image": num_rois_per_image,
        }