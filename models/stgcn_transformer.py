import torch
import torch.nn as nn
from models.stgcn import STGCN, build_adjacency
from models.transformer import TemporalTransformer

class STGCNTransformer(nn.Module):
    def __init__(self, num_classes, in_channels=3, trans_dim=256, n_layers=2, n_heads=4):
        super().__init__()
        A = build_adjacency()
        self.backbone = STGCN(in_channels=in_channels, num_classes=num_classes, A=A)
        self.proj = nn.Conv2d(256, trans_dim, kernel_size=1)
        self.transformer = TemporalTransformer(in_dim=trans_dim, n_layers=n_layers, n_heads=n_heads)
        self.classifier = nn.Linear(trans_dim, num_classes)

    def forward(self, x, mask=None):
        # x: [N, C, T, V]
        feat_map, _ = self.backbone(x)  # [N, 256, T', V]
        # Global spatial pooling to get temporal features
        feat_seq = feat_map.mean(dim=-1)  # [N, 256, T']
        feat_seq = self.proj(feat_seq.unsqueeze(-1)).squeeze(-1)  # [N, trans_dim, T']
        feat_seq = feat_seq.permute(0, 2, 1)  # [N, T', trans_dim]
        enc = self.transformer(feat_seq, mask=mask)  # [N, T', D]
        # Take cls via mean pooling (or last)
        cls_token = enc.mean(dim=1)  # [N, D]
        logits = self.classifier(cls_token)
        return logits
