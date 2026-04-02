import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    return canvas, r, (left, top)

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh.T
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return np.stack([x1,y1,x2,y2], axis=1)

def nms_np(boxes, scores, iou=0.45):
    if len(boxes) == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]

        # IoU
        x1 = np.maximum(boxes[i,0], boxes[rest,0])
        y1 = np.maximum(boxes[i,1], boxes[rest,1])
        x2 = np.minimum(boxes[i,2], boxes[rest,2])
        y2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)

        area_i = np.maximum(0, boxes[i,2]-boxes[i,0]) * np.maximum(0, boxes[i,3]-boxes[i,1])
        area_r = np.maximum(0, boxes[rest,2]-boxes[rest,0]) * np.maximum(0, boxes[rest,3]-boxes[rest,1])
        union = area_i + area_r - inter + 1e-9
        iou_vals = inter / union

        idxs = rest[iou_vals <= iou]
    return keep

def main():
    weights = os.environ["YOLO_WEIGHTS"]
    img_path = os.environ["IMG"]
    conf_thres = float(os.environ.get("CONF", "0.25"))
    iou_thres = float(os.environ.get("IOU", "0.45"))
    topk = int(os.environ.get("TOPK", "10"))

    y = YOLO(weights)
    core = y.model
    core.eval()

    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, img_path

    img_lb, r, pad = letterbox(img_bgr, (640,640))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0)

    with torch.no_grad():
        out = core(x)

    pred = out[0]  # (1, 5, 8400)
    p = pred.permute(0,2,1)[0].cpu().numpy()  # (8400,5)

    xywh = p[:, :4]
    scores = p[:, 4]

    print("raw xywh min/max:", xywh.min(axis=0), xywh.max(axis=0))
    print("raw score min/max:", scores.min(), scores.max())

    keep = scores >= conf_thres
    xywh = xywh[keep]
    scores = scores[keep]
    print("after conf filter:", len(scores))

    boxes = xywh_to_xyxy(xywh)

    # clip to 640 range for visualization safety
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, 639)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, 639)

    keep_idx = nms_np(boxes, scores, iou=iou_thres)
    keep_idx = keep_idx[:topk]
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]

    # draw + also show crops
    vis = img_lb.copy()
    for i,(b,s) in enumerate(zip(boxes, scores)):
        x1,y1,x2,y2 = b.astype(int)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"{s:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        crop = img_lb[y1:y2, x1:x2]
        if crop.size > 0:
            plt.figure(figsize=(3,3))
            plt.title(f"crop {i} score={s:.2f}")
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            plt.axis("off")

    plt.figure(figsize=(10,6))
    plt.title("detections on letterboxed image")
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()