# Inspect the raw output format returned by the Ultralytics YOLO model.
# Useful for verifying tensor/list/tuple structure before downstream decoding.

import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO

def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    canvas[top:top+nh, left:left+nw] = im_resized
    return canvas

def main():
    weights = os.environ.get("YOLO_WEIGHTS", "best.pt")
    img_path = os.environ.get("IMG", None)
    assert img_path is not None and os.path.isfile(img_path), "set IMG=/path/to/image"

    y = YOLO(weights)
    core = y.model
    core.eval()

    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None
    img_lb = letterbox(img_bgr, (640,640))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0)

    with torch.no_grad():
        out = core(x)

    print("type(out):", type(out))
    if isinstance(out, (list, tuple)):
        print("len(out):", len(out))
        for i, o in enumerate(out):
            if torch.is_tensor(o):
                print(f" out[{i}] tensor shape={tuple(o.shape)} dtype={o.dtype}")
            else:
                print(f" out[{i}] type={type(o)}")
    elif torch.is_tensor(out):
        print("out tensor shape:", tuple(out.shape), "dtype:", out.dtype)
    else:
        print("out:", out)

if __name__ == "__main__":
    main()