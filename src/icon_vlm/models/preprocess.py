# bbox/ROI가 틀어지는 문제를 피하기 위해 학습/추론 모두 letterbox 좌표계를 사용
# 원본 bbox는 여기서 letterbox bbox로 변환
from typing import Tuple
import cv2
import numpy as np

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
):
    """
    YOLO 방식 letterbox: 비율 유지 resize + padding
    return:
      img_lb: letterboxed image (new_shape)
      ratio: (r_w, r_h) - 실제론 동일 r
      pad: (dw, dh) left/top padding
    """
    shape = img.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = (r, r)
    pad = (left, top)
    return img, ratio, pad


def xyxy_orig_to_letterbox(boxes_xyxy, ratio, pad):
    """
    boxes_xyxy: (N,4) in original coords
    ratio: (r,r)
    pad: (padx, pady)
    returns: (N,4) in letterbox coords
    """
    r = ratio[0]
    padx, pady = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + padx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + pady
    return boxes


def xyxy_letterbox_to_orig(boxes_xyxy, ratio, pad):
    r = ratio[0]
    padx, pady = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padx) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pady) / r
    return boxes