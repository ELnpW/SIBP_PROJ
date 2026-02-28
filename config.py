from pathlib import Path


def get_config():
    return {
        # Training
        "batch_size": 64,
        "num_epochs": 25,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "seed": 561,

        # Model / data
        "context_size": 256,          # povecano (tekst je i dalje kompaktan)
        "model_dimension": 192,
        "num_blocks": 6,
        "heads": 6,
        "ff_multiplier": 4,
        "dropout": 0.15,
        "feature_hidden": 128,
        "num_classes": 3,

        # Loss
        "label_smoothing": 0.02,

        # LR schedule
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.15,

        # Early stopping
        "patience": 6,

        # Paths
        "train_path": "data_out/train.jsonl",
        "val_path": "data_out/val.jsonl",
        "test_path": "data_out/test.jsonl",
        "tokenizer_path": "tokenizer_out/tokenizer_football.json",
        "feature_stats_path": "data_out/feature_stats.json",

        # Saving
        "model_folder": "weights",
        "model_name": "football_transformer_hybrid.pt",

        # Logging
        "logs_dir": "logs",
    }


def get_model_path(config):
    return str(Path(config["model_folder"]) / config["model_name"])