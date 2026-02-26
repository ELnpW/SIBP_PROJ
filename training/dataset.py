import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class FootballDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer_path: str,
        context_size: int
    ):
        self.samples = []
        self.context_size = context_size

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")

        # Load data
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]

        # Tokenize
        token_ids = self.tokenizer.encode(text).ids

        # Add SOS and EOS
        token_ids = [self.sos_id] + token_ids + [self.eos_id]

        # Truncate if too long
        token_ids = token_ids[:self.context_size]

        # Pad if too short
        padding_length = self.context_size - len(token_ids)
        if padding_length > 0:
            token_ids += [self.pad_id] * padding_length

        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # Attention mask (1 = real tokenizer, 0 = pad)
        attention_mask = (input_ids != self.pad_id).unsqueeze(0).unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }