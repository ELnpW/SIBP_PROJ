import json
from pathlib import Path
from typing import Optional

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

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

        if self.pad_id is None or self.cls_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must contain [PAD], [CLS], [EOS] tokens.")

        self.samples = []
        with self.json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj or "label" not in obj or "features" not in obj:
                    raise ValueError("Each sample must contain: text, features, label.")
                self.samples.append(obj)

        # feature normalizer (set from train.py)
        self._feat_mean: Optional[torch.Tensor] = None
        self._feat_std: Optional[torch.Tensor] = None

        # sanity
        if len(self.samples) > 0:
            f0 = self.samples[0]["features"]
            if not isinstance(f0, list) or len(f0) == 0:
                raise ValueError("features must be a non-empty list[float].")

    @property
    def num_features(self) -> int:
        return len(self.samples[0]["features"])

    def set_feature_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        # mean/std: [F]
        self._feat_mean = mean.detach().clone()
        self._feat_std = std.detach().clone()

    def _normalize_features(self, feats: torch.Tensor) -> torch.Tensor:
        if self._feat_mean is None or self._feat_std is None:
            return feats
        return (feats - self._feat_mean) / self._feat_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        text = sample["text"]
        label = int(sample["label"])
        feats_list = sample["features"]

        # Tokenizer already adds [CLS] ... [EOS] because of TemplateProcessing
        ids = self.tokenizer.encode(text).ids

        # truncate/pad to context_size
        ids = ids[: self.context_size]
        if len(ids) < self.context_size:
            ids = ids + [self.pad_id] * (self.context_size - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).to(torch.long)

        feats = torch.tensor(feats_list, dtype=torch.float32)
        feats = self._normalize_features(feats)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "features": feats,
            "label": torch.tensor(label, dtype=torch.long),
        }