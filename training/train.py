import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, get_model_path
from models.football_model import FootballTransformerClassifier
from training.dataset import FootballDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_class_weights(train_dataset, num_classes: int, device: torch.device):
    """
    Robust class weights:
    - If some class has 0 samples, weight is set to 0 (so it doesn't explode to inf).
    - Also prints counts so you can verify.
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    for s in train_dataset.samples:
        y = int(s["label"])
        if y < 0 or y >= num_classes:
            raise ValueError(
                f"Found label={y} but num_classes={num_classes}. Fix config or data."
            )
        counts[y] += 1

    total = counts.sum().item()
    if total == 0:
        raise ValueError("Training set is empty (0 samples).")

    weights = torch.zeros_like(counts)

    # Standard inverse-frequency weights for classes that exist
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = counts.sum() / (num_classes * counts[i])
        else:
            weights[i] = 0.0  # IMPORTANT: avoid inf

    return counts.to(device), weights.to(device)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "all_labels": all_labels,
        "all_preds": all_preds,
    }


def pred_distribution(preds: list[int]) -> dict[int, int]:
    d: dict[int, int] = {}
    for p in preds:
        d[p] = d.get(p, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: kv[0]))


def train():
    config = get_config()
    set_seed(config["seed"])

    device = get_device()
    print("Using device:", device)

    logs_dir = Path(config.get("logs_dir", "logs"))
    logs_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = logs_dir / "metrics.csv"
    test_report_txt = logs_dir / f"test_report_{run_id}.txt"

    train_dataset = FootballDataset(
        json_path=config["train_path"],
        tokenizer_path=config["tokenizer_path"],
        context_size=config["context_size"],
    )
    val_dataset = FootballDataset(
        json_path=config["val_path"],
        tokenizer_path=config["tokenizer_path"],
        context_size=config["context_size"],
    )
    test_dataset = FootballDataset(
        json_path=config["test_path"],
        tokenizer_path=config["tokenizer_path"],
        context_size=config["context_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    vocab_size = train_dataset.tokenizer.get_vocab_size()

    # ---- Sanity check: labels must be within [0, num_classes-1]
    max_label = max(int(s["label"]) for s in train_dataset.samples)
    if max_label >= config["num_classes"]:
        raise ValueError(
            f"Config num_classes={config['num_classes']} but found label={max_label} in train set."
        )

    model = FootballTransformerClassifier(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        model_dimension=config["model_dimension"],
        num_classes=config["num_classes"],
        num_blocks=config["num_blocks"],
        heads=config["heads"],
        ff_multiplier=config["ff_multiplier"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    counts, weights = compute_class_weights(train_dataset, config["num_classes"], device)
    print("Train label counts:", counts.tolist())
    print("Class weights:", [round(float(w), 4) for w in weights.tolist()])

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_macro_f1 = -1.0
    patience = 6
    early_counter = 0

    header = [
        "run_id",
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "lr",
    ]

    # Ensure CSV has header
    write_header = not metrics_csv.exists()
    if write_header:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # If NaN happens, stop immediately with clear error
            if torch.isnan(loss):
                raise RuntimeError(
                    "Loss became NaN. Check class weights / num_classes / inputs."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        avg_loss = total_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["macro_f1"])
        lr_now = optimizer.param_groups[0]["lr"]

        pdist = pred_distribution(val_metrics["all_preds"])

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
        print(f"Pred dist (val): {pdist}")
        print(f"LR: {lr_now:.6g}\n")

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    run_id,
                    epoch + 1,
                    round(avg_loss, 6),
                    round(val_metrics["accuracy"], 6),
                    round(val_metrics["balanced_accuracy"], 6),
                    round(val_metrics["macro_f1"], 6),
                    round(val_metrics["weighted_f1"], 6),
                    lr_now,
                ]
            )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            early_counter = 0
            Path(config["model_folder"]).mkdir(exist_ok=True)
            torch.save(model.state_dict(), get_model_path(config))
            print("Best model saved.\n")
        else:
            early_counter += 1

        if early_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best checkpoint for test
    model_path = get_model_path(config)
    if Path(model_path).exists():
        print(f"Loading best checkpoint from: {model_path} (best val macro f1 = {best_macro_f1:.4f})")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("WARNING: No checkpoint found, evaluating current model.")

    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    cm = confusion_matrix(test_metrics["all_labels"], test_metrics["all_preds"])
    report = classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=["HOME_WIN", "DRAW", "AWAY_WIN"],
        digits=4,
        zero_division=0,
    )

    print("Test Accuracy:", round(test_metrics["accuracy"], 4))
    print("Test Macro F1:", round(test_metrics["macro_f1"], 4))
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    with test_report_txt.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved metrics CSV -> {metrics_csv}")
    print(f"Saved test report -> {test_report_txt}")


if __name__ == "__main__":
    train()