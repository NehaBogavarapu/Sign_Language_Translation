import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, in_dim, n_layers=2, n_heads=4, dim_ff=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        # x: [N, T, D]
        return self.encoder(x, src_key_padding_mask=mask)  # [N, T, D]
