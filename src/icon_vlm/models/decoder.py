import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TinyTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, nhead=4, num_layers=3, ffn=1024, max_len=24, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = PosEnc(d_model, max_len=max_len)

        # ROI token channel C가 런타임에 결정되므로 LazyLinear 사용
        self.mem_proj = nn.LazyLinear(d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn, dropout=dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def _causal(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward_train(self, mem_tokens: torch.Tensor, input_ids: torch.Tensor, target_ids: torch.Tensor, pad_id: int):
        mem = self.mem_proj(mem_tokens)                 # [R, Tm, D]
        x = self.pos(self.tok(input_ids))               # [R, T, D]
        T = x.size(1)
        dec = self.dec(tgt=x, memory=mem, tgt_mask=self._causal(T, x.device))
        logits = self.head(dec)                         # [R, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=0.1,   # 추가 (0.05~0.15 범위, 우선 0.1 시도)
        )
        return logits, loss

    @torch.no_grad()
    def forward_greedy(self, mem_tokens: torch.Tensor, bos_id: int, eos_id: int, pad_id: int):
        mem = self.mem_proj(mem_tokens)
        R = mem.size(0)

        out = torch.full((R, self.max_len), pad_id, dtype=torch.long, device=mem.device)
        out[:, 0] = bos_id

        finished = torch.zeros((R,), dtype=torch.bool, device=mem.device)
        for t in range(1, self.max_len):
            x = self.pos(self.tok(out[:, :t]))
            dec = self.dec(tgt=x, memory=mem, tgt_mask=self._causal(t, x.device))
            nxt = torch.argmax(self.head(dec[:, -1, :]), dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            out[:, t] = nxt
            finished = finished | (nxt == eos_id)
            if bool(finished.all()):
                break
        return out