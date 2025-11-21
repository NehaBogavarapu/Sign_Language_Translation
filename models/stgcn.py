import torch
import torch.nn as nn
import torch.nn.functional as F

def build_adjacency():
    V = 56 #68
    A = torch.zeros(V, V)

    # Index mapping
    # Pose 0-5
    POSE = list(range(0, 6))
    # Left hand 6-26
    LH = list(range(6, 27))
    # Right hand 27-47
    RH = list(range(27, 48))
    # Mouth 48-67
    MOUTH = list(range(48, 56)) #68))

    def connect(i, j):
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Pose edges: L shoulder(0)-elbow(1)-wrist(2), R shoulder(3)-elbow(4)-wrist(5), shoulders(0-3)
    connect(0,1); connect(1,2)
    connect(3,4); connect(4,5)
    connect(0,3)

    # Hand kinematic tree (MediaPipe)
    # We model a star from wrist to all fingers plus finger chains
    # Left wrist = LH[0], Right wrist = RH[0]
    def hand_edges(base):
        wrist = base
        # finger base indices relative to MediaPipe:
        # thumb: 1-4, index: 5-8, middle: 9-12, ring: 13-16, pinky: 17-20
        fingers = [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16],
            [17,18,19,20]
        ]
        for chain in fingers:
            connect(wrist, base + chain[0])  # wrist to first joint
            for i in range(len(chain)-1):
                connect(base + chain[i], base + chain[i+1])

    hand_edges(LH[0])
    hand_edges(RH[0])

    # Mouth ring: connect sequentially
    for i in range(len(MOUTH)-1):
        connect(MOUTH[i], MOUTH[i+1])
    connect(MOUTH[0], MOUTH[-1])

    # Cross-links: wrists to mouth corners (use first and a middle as corners): to help the network capture relationships between parts that are far apart spatially but may be correlated in motion
    connect(2, MOUTH[0])   # left wrist -> left mouth corner approx
    connect(5, MOUTH[4])  # right wrist -> right mouth corner approx

    # Self connections
    for i in range(V):
        A[i, i] = 1.0

    # Normalize adjacency
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D + 1e-6, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), stride=(stride,1), padding=(4,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.residual = nn.Sequential()
        if (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(stride,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [N, C, T, V]
        # Graph conv via adjacency
        x_g = torch.einsum('nctv,vw->nctw', x, self.A)
        x_g = self.gcn(x_g)
        x_t = self.tcn(x_g)
        x_res = self.residual(x)
        out = self.relu(self.bn(x_t + x_res))
        return out

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, A=None):
        super().__init__()
        if A is None:
            A = build_adjacency()
        self.A = A
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=2),
            STGCNBlock(128, 256, A, stride=2),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [N, C, T, V]
        N, C, T, V = x.shape
        x = x.reshape(N, C*V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, T, V)
        for layer in self.layers:
            x = layer(x)
        # Return both pooled and per-time features
        pooled = self.pool(x).squeeze(-1).squeeze(-1)  # [N, 256]
        return x, pooled  # [N, C', T', V], [N, 256]
