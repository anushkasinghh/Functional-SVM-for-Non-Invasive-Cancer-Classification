# evaluate.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap

@torch.no_grad()
def compute_invariance(encoder, X, augmentor, device='cpu', n_samples=200):
    encoder.eval()
    M = X.shape[0]
    idx = np.random.choice(M, size=min(n_samples, M), replace=False)
    x_sel = X[idx]
    x_aug, _, _ = augmentor.sample(x_sel)
    z_orig, _ = encoder(torch.tensor(x_sel, dtype=torch.float32).to(device))
    z_aug, _ = encoder(torch.tensor(x_aug, dtype=torch.float32).to(device))
    z_orig = F.normalize(z_orig, dim=1)
    z_aug  = F.normalize(z_aug, dim=1)
    pos_dists = torch.norm(z_orig - z_aug, dim=1).cpu().numpy()
    rand_idx = np.random.choice(M, size=len(idx), replace=True)
    x_rand = X[rand_idx]
    z_rand, _ = encoder(torch.tensor(x_rand, dtype=torch.float32).to(device))
    z_rand = F.normalize(z_rand, dim=1)
    neg_dists = torch.norm(z_orig - z_rand, dim=1).cpu().numpy()
    return pos_dists, neg_dists

@torch.no_grad()
def plot_umap(encoder, X, labels, augmentor=None, device='cpu', apply_aug=False):
    encoder.eval()
    xs = X.copy()
    if apply_aug:
        xs,_,_ = augmentor.sample(xs)
    zs, _ = encoder(torch.tensor(xs, dtype=torch.float32).to(device))
    Z = zs.cpu().numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    Z2 = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    for c in np.unique(labels):
        mask = (labels == c)
        plt.scatter(Z2[mask,0], Z2[mask,1], label=f"class{c}", s=8, alpha=0.8)
    plt.legend(); plt.title("UMAP embeddings")
    plt.show()
