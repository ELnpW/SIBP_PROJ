import csv
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


def compute_class_weights(train_dataset: FootballDataset, num_classes: int, device):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for s in train_dataset.samples:
        counts[int(s["label"])] += 1
    weights = (counts.sum() / (num_classes * counts)).to(device)
    return counts, weights


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

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "all_labels": all_labels,
        "all_preds": all_preds,
    }


def write_csv_row(csv_path: Path, header: list[str], row: dict):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def train():
    config = get_config()
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = logs_dir / "metrics.csv"
    test_report_txt = logs_dir / f"test_report_{run_id}.txt"

    # Datasets
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

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    vocab_size = train_dataset.tokenizer.get_vocab_size()

    # Model
    model = FootballTransformerClassifier(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        model_dimension=config["model_dimension"],
        num_classes=config["num_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Weighted loss
    counts, weights = compute_class_weights(train_dataset, config["num_classes"], device)
    print("Train label counts:", counts.tolist())
    print("Class weights:", [round(w, 4) for w in weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_macro_f1 = 0.0
    header = [
        "run_id",
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
    ]

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}\n")

        # Log metrics to CSV
        write_csv_row(
            metrics_csv,
            header,
            {
                "run_id": run_id,
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
                "val_macro_f1": round(val_metrics["macro_f1"], 6),
                "val_weighted_f1": round(val_metrics["weighted_f1"], 6),
            },
        )

        # Save best model by Macro F1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            Path(config["model_folder"]).mkdir(exist_ok=True)
            torch.save(model.state_dict(), get_model_path(config))
            print("Best model saved.\n")

    # Final test evaluation + report
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    cm = confusion_matrix(test_metrics["all_labels"], test_metrics["all_preds"])
    report = classification_report(
        test_metrics["all_labels"],
        test_metrics["all_preds"],
        target_names=["HOME_WIN", "DRAW", "AWAY_WIN"],
        digits=4,
    )

    print("Test Accuracy:", round(test_metrics["accuracy"], 4))
    print("Test Balanced Acc:", round(test_metrics["balanced_accuracy"], 4))
    print("Test Macro F1:", round(test_metrics["macro_f1"], 4))
    print("Test Weighted F1:", round(test_metrics["weighted_f1"], 4))

    with test_report_txt.open("w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write("=== Config ===\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Test metrics ===\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.6f}\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.6f}\n")
        f.write("\n=== Confusion matrix ===\n")
        f.write(np.array2string(cm))
        f.write("\n\n=== Classification report ===\n")
        f.write(report)

    print(f"\nSaved metrics CSV -> {metrics_csv}")
    print(f"Saved test report -> {test_report_txt}")


if __name__ == "__main__":
    train()