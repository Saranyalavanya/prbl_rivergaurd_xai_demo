"""Train the Siamese change-detection model.

Default: trains on procedurally generated river scenes (no download needed).
Usage:  python train.py [--epochs 15] [--samples 240] [--size 128]
To use REAL labelled data (e.g. OSCD / LEVIR-CD), replace `SceneDataset` with your
own Dataset returning (before[4,H,W], after[4,H,W], mask[H,W]) float32 tensors.
"""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from riverguard.config import MODEL_PATH
from riverguard.model import SiameseChangeNet, to_tensor
from riverguard.synthetic import make_scene
from riverguard.preprocess import preprocess_pair


class SceneDataset(Dataset):
    def __init__(self, n, size, seed0):
        self.n, self.size, self.seed0 = n, size, seed0

    def __len__(self): return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed0 + i)
        b, a, t = make_scene(self.size, seed=self.seed0 + i, mining=rng.random() > 0.15)
        b, a = preprocess_pair(b, a)
        k = int(rng.integers(0, 4))
        b, a, t = (np.rot90(x, k, axes=(-2, -1)) for x in (b, a, t))
        if rng.random() < .5:
            b, a, t = (np.flip(x, -1) for x in (b, a, t))
        return (to_tensor(np.ascontiguousarray(b))[0], to_tensor(np.ascontiguousarray(a))[0],
                torch.from_numpy(np.ascontiguousarray(t))[None])


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def train(epochs=15, samples=240, size=128, batch=8, lr=2e-3, verbose=True, progress=None):
    torch.manual_seed(0)
    n_val = max(16, samples // 8)
    tr = DataLoader(SceneDataset(samples, size, 1000), batch_size=batch, shuffle=True)
    va = DataLoader(SceneDataset(n_val, size, 900000), batch_size=batch)
    model = SiameseChangeNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(tr))
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tot = 0
        for x1, x2, y in tr:
            logit = model(x1, x2)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(4.0)) + dice_loss(logit, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval(); tp = fp = fn = 0
        with torch.no_grad():
            for x1, x2, y in va:
                p = (torch.sigmoid(model(x1, x2)) > .5).float()
                tp += (p * y).sum().item(); fp += (p * (1 - y)).sum().item(); fn += ((1 - p) * y).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        if verbose:
            print(f"epoch {ep+1:02d}/{epochs}  loss {tot/len(tr):.3f}  val-F1 {f1:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if progress: progress((ep + 1) / epochs, f1)
    torch.save(model.state_dict(), MODEL_PATH)
    if verbose: print("Saved model ->", MODEL_PATH)
    return f1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--samples", type=int, default=240)
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    train(a.epochs, a.samples, a.size)
