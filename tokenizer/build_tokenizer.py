import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

DATA_PATH = Path("data_out/train.jsonl")
TOKENIZER_PATH = Path("tokenizer_out/tokenizer_football.json")
TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("tokenizer_out")

def load_texts():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["text"]

def main():
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=2
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.train_from_iterator(load_texts(), trainer)
    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer saved.")
    print("Vocab size:", tokenizer.get_vocab_size())

if __name__ == "__main__":
    main()