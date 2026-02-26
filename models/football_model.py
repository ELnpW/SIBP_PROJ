import torch
import torch.nn as nn

from model import (
    Encoder,
    EncoderBlock,
    MultiHeadAttentionBlock,
    FeedForwardBlock,
    InputEmbeddings,
    PositionalEncoding
)

class FootballTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        model_dimension: int = 128,
        number_of_blocks: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        feed_forward_dimension: int = 256,
        num_classes: int = 3
    ):
        super().__init__()

        # Embedding
        self.embedding = InputEmbeddings(model_dimension, vocab_size)
        self.position = PositionalEncoding(model_dimension, context_size, dropout)

        # Encoder blocks
        encoder_blocks = []
        for _ in range(number_of_blocks):
            self_attention = MultiHeadAttentionBlock(model_dimension, heads, dropout)
            feed_forward = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
            encoder_blocks.append(
                EncoderBlock(model_dimension, self_attention, feed_forward, dropout)
            )

        self.encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))

        # Classification head
        self.classifier = nn.Linear(model_dimension, num_classes)

    def forward(self, input_ids, attention_mask):
        # (batch, seq_len)
        x = self.embedding(input_ids)
        x = self.position(x)

        encoder_output = self.encoder(x, attention_mask)

        # Uzmi reprezentaciju prvog tokena
        cls_representation = encoder_output[:, 0, :]  # (batch, d_model)

        logits = self.classifier(cls_representation)  # (batch, 3)

        return logits