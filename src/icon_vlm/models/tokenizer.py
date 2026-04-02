import string
from dataclasses import dataclass
from typing import List

@dataclass
class CharTokenizer:
    pad: str = "<PAD>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"

    def __post_init__(self):
        alphabet = list(string.ascii_lowercase + string.digits + "_- ")
        self.itos = [self.pad, self.bos, self.eos] + alphabet
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad]  # 0
        self.bos_id = self.stoi[self.bos]  # 1
        self.eos_id = self.stoi[self.eos]  # 2
        self.vocab_size = len(self.itos)

    def encode(self, s: str, max_len: int) -> List[int]:
        s = s.lower()
        ids = [self.bos_id]
        for ch in s:
            ids.append(self.stoi.get(ch, self.stoi["_"]))
        ids.append(self.eos_id)

        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for t in ids:
            if t == self.eos_id:
                break
            if t in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[t])
        return "".join(out).strip()