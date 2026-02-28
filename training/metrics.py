from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)


@dataclass
class EvalResult:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    all_labels: List[int]
    all_preds: List[int]


@torch.no_grad()
def evaluate(model, dataloader, device) -> EvalResult:
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, features)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return EvalResult(
        accuracy=float(accuracy_score(all_labels, all_preds)),
        balanced_accuracy=float(balanced_accuracy_score(all_labels, all_preds)),
        macro_f1=float(f1_score(all_labels, all_preds, average="macro")),
        weighted_f1=float(f1_score(all_labels, all_preds, average="weighted")),
        all_labels=all_labels,
        all_preds=all_preds,
    )


def pred_distribution(preds: List[int]) -> Dict[int, int]:
    d: Dict[int, int] = {}
    for p in preds:
        d[p] = d.get(p, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: kv[0]))


def make_reports(labels: List[int], preds: List[int]) -> dict:
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=["HOME_WIN", "DRAW", "AWAY_WIN"],
        digits=4,
        zero_division=0,
    )
    return {"confusion_matrix": cm, "classification_report": report}