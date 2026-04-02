import torch
from tqdm import tqdm

# torch 2.x 권장 AMP API
from torch.amp import autocast, GradScaler

def _get_device_type(device: str) -> str:
    # 'cuda', 'cpu', 'mps' 등을 반환
    if device is None:
        return "cpu"
    d = str(device)
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


@torch.no_grad()
def token_accuracy_from_logits(logits: torch.Tensor, gt_text_ids: torch.Tensor, pad_id: int) -> float:
    """
    logits: [R, L-1, V]
    gt_text_ids: [R, L]  (BOS ... EOS/PAD)
    pad_id: tokenizer.pad_id

    return: token accuracy (PAD 제외)
    """
    # target: 다음 토큰
    target = gt_text_ids[:, 1:]                # [R, L-1]
    pred = logits.argmax(dim=-1)               # [R, L-1]

    mask = target.ne(pad_id)                   # PAD 제외
    denom = mask.sum().item()
    if denom == 0:
        return 0.0

    correct = (pred.eq(target) & mask).sum().item()
    return float(correct) / float(denom)


def train_one_epoch(model, loader, optimizer, device, amp: bool = True, log_every: int = 10):
    model.train()

    device_type = _get_device_type(device)
    amp = bool(amp) and (device_type == "cuda")  # CUDA 아닐 땐 AMP 끔 (경고/무의미 방지)

    scaler = GradScaler(enabled=amp)

    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, (images, gt_boxes_list, gt_text_ids) in pbar:
        images = images.to(device)
        gt_text_ids = gt_text_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=amp):
            out = model.forward_train(images, gt_boxes_list, gt_text_ids)
            loss = out["loss"]
            logits = out.get("logits", None)

        # backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # stats
        total_loss += float(loss.detach().cpu())
        if logits is not None:
            acc = token_accuracy_from_logits(logits.detach(), gt_text_ids, pad_id=model.tokenizer.pad_id)
            total_acc += acc

        n_steps += 1

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / max(1, n_steps)
            avg_acc = total_acc / max(1, n_steps)
            pbar.set_description(f"loss={avg_loss:.4f} tok_acc={avg_acc:.3f}")

    return {
        "loss": total_loss / max(1, n_steps),
        "token_acc": total_acc / max(1, n_steps),
    }

@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    amp: bool = True,
    log_every: int = 10,
    *,
    log_samples: int = 0,
    sample_step: int = 0,
):
    """
    log_samples: 0이면 비활성. >0이면 sample_step에서 배치 내 랜덤 N개 GT/PRED 디코딩 출력
    sample_step: 몇 번째 step에서 샘플을 찍을지 (기본 0: 첫 배치)
    """
    import random

    model.eval()

    device_type = _get_device_type(device)
    amp = bool(amp) and (device_type == "cuda")  # eval도 동일하게 CUDA 아니면 끔

    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0

    printed = False  # 샘플 한 번만 출력

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, (images, gt_boxes_list, gt_text_ids) in pbar:
        images = images.to(device)
        gt_text_ids = gt_text_ids.to(device)

        with autocast(device_type=device_type, enabled=amp):
            out = model.forward_train(images, gt_boxes_list, gt_text_ids)
            loss = out["loss"]
            logits = out.get("logits", None)

        total_loss += float(loss.detach().cpu())
        if logits is not None:
            acc = token_accuracy_from_logits(logits.detach(), gt_text_ids, pad_id=model.tokenizer.pad_id)
            total_acc += acc

            # -------- 샘플 디코딩 로그 (개선: 진짜 greedy decode) --------
            if (not printed) and (log_samples > 0) and (step == sample_step):
                printed = True

                dbg = model.forward_train_decode(
                    images=images,
                    gt_boxes_lb_list=gt_boxes_list,
                    device=device,
                    max_len=gt_text_ids.size(1),
                )
                pred_texts_all = dbg["pred_texts"]  # 길이 R

                # GT text도 decode
                decode = model.tokenizer.decode
                gt_texts_all = [decode(row.tolist()) for row in gt_text_ids]

                R = len(gt_texts_all)
                idxs = list(range(R))
                random.shuffle(idxs)
                idxs = idxs[: min(log_samples, R)]

                print("\n" + "=" * 80)
                print(f"[VAL SAMPLES - GREEDY] step={step} (show {len(idxs)}/{R})")
                for i, r in enumerate(idxs):
                    print(f"\n[{i}]")
                    print(f"GT  : {gt_texts_all[r]}")
                    print(f"PRED: {pred_texts_all[r]}")
                print("=" * 80 + "\n")
            # ------------------------------------------------------------

        n_steps += 1

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / max(1, n_steps)
            avg_acc = total_acc / max(1, n_steps)
            pbar.set_description(f"val_loss={avg_loss:.4f} val_tok_acc={avg_acc:.3f}")

    return {
        "loss": total_loss / max(1, n_steps),
        "token_acc": total_acc / max(1, n_steps),
    }