"""A shared, one-layer transformer for two ten-match team histories."""

import math

import torch
from torch import nn


class MatchTransformer(nn.Module):
    def __init__(self, positional_encoding="none", history_length=10,
                 input_features=3, d_model=32, nhead=2, dim_feedforward=64,
                 dropout=0.1, num_classes=3):
        super().__init__()
        if positional_encoding not in ("none", "sinusoidal"):
            raise ValueError("positional_encoding must be 'none' or 'sinusoidal'")
        if d_model % 2:
            raise ValueError("Sinusoidal encoding requires an even d_model")
        self.config = {
            "positional_encoding": positional_encoding,
            "history_length": history_length, "input_features": input_features,
            "d_model": d_model, "nhead": nhead,
            "dim_feedforward": dim_feedforward, "dropout": dropout,
            "num_classes": num_classes,
        }
        self.projection = nn.Linear(input_features, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.classifier = nn.Linear(2 * d_model, num_classes)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model)); odd channels use cos.
        position = torch.arange(history_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(history_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("positions", encoding.unsqueeze(0))

    def encode_team(self, history):
        expected = (self.config["history_length"], self.config["input_features"])
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected[0]}, {expected[1]}] history")
        embeddings = self.projection(history)
        if self.config["positional_encoding"] == "sinusoidal":
            embeddings = embeddings + self.positions
        # All ten known results may attend to each other: no causal mask.
        return self.encoder(embeddings).mean(dim=1)

    def forward(self, home_history, away_history):
        # The same projection and encoder are called for both teams.
        home = self.encode_team(home_history)
        away = self.encode_team(away_history)
        return self.classifier(torch.cat((home, away), dim=1))
