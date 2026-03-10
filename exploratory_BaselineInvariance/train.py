# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from models import nt_xent

def train_contrastive(encoder, X, augmentor, epochs=20, batch_size=128, lr=1e-3, device='cpu'):
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    encoder.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, in loader:
            xb = xb.to(device)
            x1, _, _ = augmentor.sample(xb.cpu().numpy())
            x2, _, _ = augmentor.sample(xb.cpu().numpy())
            x1 = torch.tensor(x1, dtype=torch.float32).to(device)
            x2 = torch.tensor(x2, dtype=torch.float32).to(device)
            _, p1 = encoder(x1)
            _, p2 = encoder(x2)
            loss = nt_xent(p1, p2, temperature=0.2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs}, loss={total_loss/len(loader):.4f}")
    return encoder
