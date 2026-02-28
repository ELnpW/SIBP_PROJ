import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


DATA_PATH = Path("data_out/train.jsonl")
TOKENIZER_PATH = Path("tokenizer_out/tokenizer_football.json")
TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[EOS]"]


def load_texts():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["text"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run prepare_data.py first.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=16000,          # mozes 8000 ako ti je dataset manji
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(load_texts(), trainer)

    # Setup post-processor to always add [CLS] ... [EOS]
    cls_id = tokenizer.token_to_id("[CLS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if cls_id is None or eos_id is None:
        raise RuntimeError("Special tokens not found after training. Check SPECIAL_TOKENS.")

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [EOS]",
        pair="[CLS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[EOS]", eos_id),
        ],
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer saved:", TOKENIZER_PATH)
    print("Vocab size:", tokenizer.get_vocab_size())
    print("IDs:",
          {"UNK": tokenizer.token_to_id("[UNK]"),
           "PAD": tokenizer.token_to_id("[PAD]"),
           "CLS": tokenizer.token_to_id("[CLS]"),
           "EOS": tokenizer.token_to_id("[EOS]")})


if __name__ == "__main__":
    main()