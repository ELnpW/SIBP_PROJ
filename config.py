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
        "seed": 561,

        # Model
        "context_size": 64,
        "model_dimension": 128,
        "num_classes": 3,

        # Paths
        "train_path": "data_out/train.jsonl",
        "val_path": "data_out/val.jsonl",
        "test_path": "data_out/test.jsonl",
        "tokenizer_path": "tokenizer_out/tokenizer_football.json",

        # Saving
        "model_folder": "weights",
        "model_name": "football_transformer.pt",
    }


def get_model_path(config):
    """
    Returns full path where the model will be saved.
    """
    return str(Path(config["model_folder"]) / config["model_name"])