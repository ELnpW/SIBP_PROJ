import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, get_model_path
from models.football_model import HybridFootballTransformerClassifier
from training.dataset import FootballDataset
from training.metrics import evaluate, pred_distribution, make_reports


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
    counts = torch.zeros(num_classes, dtype=torch.float)
    for s in train_dataset.samples:
        y = int(s["label"])
        if y < 0 or y >= num_classes:
            raise ValueError(f"Found label={y} but num_classes={num_classes}. Fix config or data.")
        counts[y] += 1

    total = counts.sum().item()
    if total == 0:
        raise ValueError("Training set is empty (0 samples).")

    weights = torch.zeros_like(counts)
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = counts.sum() / (num_classes * counts[i])
        else:
            weights[i] = 0.0

    return counts.to(device), weights.to(device)


def compute_feature_stats(train_dataset: FootballDataset):
    feats = torch.stack(
        [torch.tensor(s["features"], dtype=torch.float32) for s in train_dataset.samples],
        dim=0
    )  # [N,F]
    mean = feats.mean(dim=0)
    std = feats.std(dim=0, unbiased=False).clamp(min=1e-6)
    return mean, std


def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """
    LR schedule:
      - linear warmup to base LR
      - cosine decay to base_lr * min_lr_ratio
    """
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    feat_stats_path = Path(config.get("feature_stats_path", "data_out/feature_stats.json"))

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

    # --- feature normalization (train stats)
    feat_mean, feat_std = compute_feature_stats(train_dataset)
    train_dataset.set_feature_normalizer(feat_mean, feat_std)
    val_dataset.set_feature_normalizer(feat_mean, feat_std)
    test_dataset.set_feature_normalizer(feat_mean, feat_std)

    feat_stats_path.write_text(
        json.dumps(
            {
                "num_features": int(train_dataset.num_features),
                "mean": feat_mean.tolist(),
                "std": feat_std.tolist(),
            },
            indent=2
        ),
        encoding="utf-8"
    )
    print("Saved feature stats ->", feat_stats_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    vocab_size = train_dataset.tokenizer.get_vocab_size()

    # Sanity: labels range
    max_label = max(int(s["label"]) for s in train_dataset.samples)
    if max_label >= config["num_classes"]:
        raise ValueError(
            f"Config num_classes={config['num_classes']} but found label={max_label} in train set."
        )

    model = HybridFootballTransformerClassifier(
        vocab_size=vocab_size,
        context_size=config["context_size"],
        model_dimension=config["model_dimension"],
        num_classes=config["num_classes"],
        num_features=train_dataset.num_features,
        num_blocks=config["num_blocks"],
        heads=config["heads"],
        ff_multiplier=config["ff_multiplier"],
        dropout=config["dropout"],
        feature_hidden=config.get("feature_hidden", 128),
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

    total_steps = config["num_epochs"] * max(1, len(train_loader))
    warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.06)))
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=float(config.get("min_lr_ratio", 0.15)),
    )

    best_macro_f1 = -1.0
    patience = int(config.get("patience", 6))
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

    write_header = not metrics_csv.exists()
    if write_header:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    global_step = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask, features)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN. Check data / features / weights / LR.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        avg_loss = total_loss / max(1, len(train_loader))

        val_res = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        pdist = pred_distribution(val_res.all_preds)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_res.accuracy:.4f}")
        print(f"Val Balanced Acc: {val_res.balanced_accuracy:.4f}")
        print(f"Val Macro F1: {val_res.macro_f1:.4f}")
        print(f"Val Weighted F1: {val_res.weighted_f1:.4f}")
        print(f"Pred dist (val): {pdist}")
        print(f"LR: {lr_now:.6g}\n")

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    run_id,
                    epoch + 1,
                    round(avg_loss, 6),
                    round(val_res.accuracy, 6),
                    round(val_res.balanced_accuracy, 6),
                    round(val_res.macro_f1, 6),
                    round(val_res.weighted_f1, 6),
                    lr_now,
                ]
            )

        if val_res.macro_f1 > best_macro_f1:
            best_macro_f1 = val_res.macro_f1
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
    test_res = evaluate(model, test_loader, device)

    print("Test Accuracy:", round(test_res.accuracy, 4))
    print("Test Macro F1:", round(test_res.macro_f1, 4))

    reports = make_reports(test_res.all_labels, test_res.all_preds)
    print("\nConfusion matrix:\n", reports["confusion_matrix"])
    print("\nClassification report:\n", reports["classification_report"])

    with test_report_txt.open("w", encoding="utf-8") as f:
        f.write(reports["classification_report"])

    print(f"\nSaved metrics CSV -> {metrics_csv}")
    print(f"Saved test report -> {test_report_txt}")


if __name__ == "__main__":
    train()