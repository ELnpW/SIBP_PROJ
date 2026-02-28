import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class FootballDataset(Dataset):
    def __init__(self, json_path: str, tokenizer_path: str, context_size: int):
        self.json_path = Path(json_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.context_size = int(context_size)

        if not self.json_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.json_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer file: {self.tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # Special tokens (your tokenizer has these IDs)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

        if self.pad_id is None or self.sos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must contain [PAD], [SOS], [EOS] tokens.")

        # Load samples
        self.samples = []
        with self.json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # expected: {"text": "...", "label": 0/1/2, ...}
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        text = sample["text"]
        label = int(sample["label"])

        ids = self.tokenizer.encode(text).ids
        ids = [self.sos_id] + ids + [self.eos_id]

        # truncate/pad
        ids = ids[: self.context_size]
        if len(ids) < self.context_size:
            ids = ids + [self.pad_id] * (self.context_size - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).to(torch.long)  # 1 real, 0 pad
        label_t = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label_t,
        }