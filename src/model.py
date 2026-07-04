"""
model.py
A basic LSTM autoencoder for trajectory imputation.

Input: masked (x, y) sequence + mask (so the model knows what's real vs missing)
Output: full reconstructed (x, y) sequence, including filled-in values for missing points
"""

import torch
import torch.nn as nn


class LSTMImputer(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout only applies between stacked layers
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, masked_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_feature = mask.unsqueeze(-1).float()
        combined_input = torch.cat([masked_seq, mask_feature], dim=-1)

        encoder_output, (hidden, cell) = self.encoder(combined_input)
        decoder_output, _ = self.decoder(encoder_output)
        reconstructed = self.output_layer(decoder_output)

        return reconstructed


if __name__ == "__main__":
    # Quick sanity check with dummy data
    batch_size, seq_len = 4, 50

    dummy_masked_seq = torch.rand(batch_size, seq_len, 2)
    dummy_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

    model = LSTMImputer(input_size=2, hidden_size=64, num_layers=1)
    output = model(dummy_masked_seq, dummy_mask)

    print(f"Input shape: {dummy_masked_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")