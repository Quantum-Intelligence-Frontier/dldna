from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch
import math

class SimpleTransformer(nn.Module):
    """A simplified Transformer model for image classification.

    This model flattens the input image, applies an encoding layer,
    adds positional encoding, processes it through a Transformer encoder,
    and then uses a decoder to produce class predictions.
    """
    def __init__(self, num_tokens=3072, num_classes=100,
                 d_model=768, nhead=12, num_layers=8, dropout=0.1):
        super().__init__()

        # Convert the input image into a sequence
        self.flatten = nn.Flatten()

        # Enhanced input embedding
        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.GELU()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Enhanced Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Use Pre-LN structure
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

        # Weight initialization
        self._init_weights()

        self.config = {
            "model_name": "SimpleTransformer",
            "d_model": d_model,
            "num_classes": num_classes,
            "num_layers": num_layers,
            "nhead": nhead
        }

    def _init_weights(self):
        """Initializes the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Performs a forward pass through the model.

        Args:
            x: The input tensor (image).

        Returns:
            The output tensor (class predictions).
        """
        x = self.flatten(x)
        x = self.encoder(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Enhanced global pooling
        x = 0.5 * x.mean(dim=1) + 0.5 * x.max(dim=1)[0]  # Combine mean and max pooling
        x = self.decoder(x)
        return x



class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models.

    Adds positional information to the input embeddings.  Uses the standard
    sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter, but persistent

    def forward(self, x):
        """Adds positional encoding to the input.

        Args:
            x: The input tensor.

        Returns:
            The input tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)