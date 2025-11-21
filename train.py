import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.sign_dataset import SignDataset
from models.stgcn_transformer import STGCNTransformer
from tqdm import tqdm

def main():
    data_dir = 'data/sequences'
    num_classes = load_num_classes()
    train_set = SignDataset(data_dir, split='train', max_len=128)
    val_set = SignDataset(data_dir, split='val', max_len=128)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCNTransformer(num_classes=num_classes, in_channels=3, trans_dim=256, n_layers=2, n_heads=4).to(device)
    criterion = nn.CrossEntropyLoss() # uses cross entropy as loss function ("widely considered a highly effective and standard loss function for classification tasks")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) # saw online that this is "a technique used in deep learning to adjust the learning rate during training, following the shape of a cosine curve"

    epochs = 1 # have to experiment with this
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}')

    file_name = 'stgcn_transformer.pth' # Update this to save new model!
    torch.save(model.state_dict(), os.path.join('checkpoints', file_name))

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return running_loss/total, correct/total

def load_num_classes():
    path = 'data/label_map.npz'
    m = np.load(path, allow_pickle=True)
    gloss_to_id = m['gloss_to_id'].item()
    return len(gloss_to_id)

if __name__ == '__main__':
    import numpy as np
    os.makedirs('checkpoints', exist_ok=True)
    main()
