from pathlib import Path


def get_config():
    """
    Configuration file for Football Transformer Classification project.
    """
    return {
        # Training
        "batch_size": 64,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "seed": 561,

        # Model / data
        "context_size": 160,
        "model_dimension": 192,
        "num_blocks": 6,
        "heads": 6,
        "ff_multiplier": 4,
        "dropout": 0.15,
        "num_classes": 3,  # !!! MUST be 3 (HOME/DRAW/AWAY)

        # Loss
        "label_smoothing": 0.02,

        # Paths
        "train_path": "data_out/train.jsonl",
        "val_path": "data_out/val.jsonl",
        "test_path": "data_out/test.jsonl",
        "tokenizer_path": "tokenizer_out/tokenizer_football.json",

        # Saving
        "model_folder": "weights",
        "model_name": "football_transformer.pt",

        # Logging
        "logs_dir": "logs",
    }


def get_model_path(config):
    """
    Returns full path where the model will be saved.
    """
    return str(Path(config["model_folder"]) / config["model_name"])