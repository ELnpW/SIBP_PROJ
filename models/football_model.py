import inspect
import torch
import torch.nn as nn

from model import (
    Encoder,
    EncoderBlock,
    MultiHeadAttentionBlock,
    FeedForwardBlock,
    InputEmbeddings,
    PositionalEncoding,
)


def _build_encoder(model_dimension: int, layers: nn.ModuleList):
    """
    Kompatibilno sa raznim implementacijama:
      - Encoder(layers)
      - Encoder(model_dimension, layers)
    """
    sig = inspect.signature(Encoder.__init__)
    n_params = len(sig.parameters)

    if n_params == 2:   # self + layers
        return Encoder(layers)

    if n_params == 3:   # self + model_dimension + layers
        return Encoder(model_dimension, layers)

    # fallback
    try:
        return Encoder(model_dimension, layers)
    except TypeError:
        return Encoder(layers)


class FootballTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        model_dimension: int,
        num_classes: int,
        num_blocks: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
    ):
        super().__init__()

        self.context_size = context_size
        self.model_dimension = model_dimension

        self.embed = InputEmbeddings(model_dimension, vocab_size)
        self.pos = PositionalEncoding(model_dimension, context_size, dropout)

        layers = []
        for _ in range(num_blocks):
            self_attn = MultiHeadAttentionBlock(model_dimension, heads, dropout)
            ff = FeedForwardBlock(model_dimension, model_dimension * ff_multiplier, dropout)

            # EncoderBlock može biti:
            # - EncoderBlock(self_attn, ff, dropout)
            # - EncoderBlock(model_dimension, self_attn, ff, dropout)
            try:
                layers.append(EncoderBlock(model_dimension, self_attn, ff, dropout))
            except TypeError:
                layers.append(EncoderBlock(self_attn, ff, dropout))

        self.encoder = _build_encoder(model_dimension, nn.ModuleList(layers))
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(model_dimension, model_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, num_classes),
        )

    def _build_source_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: [B,S] -> [B,1,1,S] za broadcast u attention
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def _mean_pool(self, enc_out: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling preko svih valid tokena (ne-pad).
        enc_out: [B,S,D]
        attention_mask: [B,S] (1 valid, 0 pad)
        """
        mask = attention_mask.unsqueeze(-1).float()          # [B,S,1]
        summed = (enc_out * mask).sum(dim=1)                 # [B,D]
        denom = mask.sum(dim=1).clamp(min=1.0)               # [B,1]
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embed(input_ids)                 # [B,S,D]
        x = self.pos(x)                           # [B,S,D]
        src_mask = self._build_source_mask(attention_mask)   # [B,1,1,S]

        enc_out = self.encoder(x, src_mask)       # [B,S,D]
        pooled = self._mean_pool(enc_out, attention_mask)    # [B,D]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)          # [B,C]
        return logits