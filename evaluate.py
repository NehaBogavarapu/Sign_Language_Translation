import os
import torch
from torch.utils.data import DataLoader
from datasets.sign_dataset import SignDataset
from models.stgcn_transformer import STGCNTransformer
import numpy as np

def main():
    data_dir = 'data/sequences'
    label_map = np.load('data/label_map.npz', allow_pickle=True)['gloss_to_id'].item()
    num_classes = len(label_map)
    model = STGCNTransformer(num_classes=num_classes)

    checkpoint_file = 'checkpoints/stgcn_transformer.pth' # update this to change the loaded file!

    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    val_set = SignDataset(data_dir, split='test', max_len=128)
    loader = DataLoader(val_set, batch_size=32, shuffle=False)
    acc = accuracy(model, loader, device)
    print(f'Test accuracy: {acc:.4f}')

def accuracy(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total

if __name__ == '__main__':
    main()
