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
    sig = inspect.signature(Encoder.__init__)
    n_params = len(sig.parameters)

    if n_params == 2:   # self + layers
        return Encoder(layers)
    if n_params == 3:   # self + model_dimension + layers
        return Encoder(model_dimension, layers)

    try:
        return Encoder(model_dimension, layers)
    except TypeError:
        return Encoder(layers)


class HybridFootballTransformerClassifier(nn.Module):
    """
    Transformer encoder za tekst + MLP za numericke feature-e.
    Pooled repr = CLS token (pozicija 0).
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        model_dimension: int,
        num_classes: int,
        num_features: int,
        num_blocks: int = 6,
        heads: int = 6,
        dropout: float = 0.15,
        ff_multiplier: int = 4,
        feature_hidden: int = 128,
    ):
        super().__init__()

        self.context_size = context_size
        self.model_dimension = model_dimension
        self.num_features = num_features

        # Text branch
        self.embed = InputEmbeddings(model_dimension, vocab_size)
        self.pos = PositionalEncoding(model_dimension, context_size, dropout)

        layers = []
        for _ in range(num_blocks):
            self_attn = MultiHeadAttentionBlock(model_dimension, heads, dropout)
            ff = FeedForwardBlock(model_dimension, model_dimension * ff_multiplier, dropout)
            try:
                layers.append(EncoderBlock(model_dimension, self_attn, ff, dropout))
            except TypeError:
                layers.append(EncoderBlock(self_attn, ff, dropout))

        self.encoder = _build_encoder(model_dimension, nn.ModuleList(layers))

        # Numeric branch
        self.feat_net = nn.Sequential(
            nn.Linear(num_features, feature_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden, model_dimension),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(model_dimension * 2)
        self.dropout = nn.Dropout(dropout)

        # Classifier over concat(text_repr, feat_repr)
        self.classifier = nn.Sequential(
            nn.Linear(model_dimension * 2, model_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension, num_classes),
        )

    def _build_source_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # [B,S] -> [B,1,1,S]
        return attention_mask.unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, features: torch.Tensor):
        # Text
        x = self.embed(input_ids)               # [B,S,D]
        x = self.pos(x)                         # [B,S,D]
        src_mask = self._build_source_mask(attention_mask)

        enc_out = self.encoder(x, src_mask)     # [B,S,D]

        # CLS pooling (assumes tokenizer adds [CLS] at position 0)
        text_repr = enc_out[:, 0, :]            # [B,D]

        # Numeric
        feat_repr = self.feat_net(features)     # [B,D]

        # Fuse
        fused = torch.cat([text_repr, feat_repr], dim=1)  # [B,2D]
        fused = self.norm(fused)
        fused = self.dropout(fused)

        logits = self.classifier(fused)         # [B,C]
        return logits