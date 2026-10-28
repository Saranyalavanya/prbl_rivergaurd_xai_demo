"""Explainable AI: Grad-CAM for the Siamese change-detection network.

Target layer  = bottleneck change features [|f1-f2|, f1, f2].
Target score  = sum of 'change' logits inside the predicted change mask.
The resulting heat-map shows which image regions drove the model's mining decision.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .model import to_tensor, _pad8


def grad_cam(model, before, after, mask):
    model.eval()
    x1, h, w = _pad8(to_tensor(before)); x2, _, _ = _pad8(to_tensor(after))
    x1.requires_grad_(True)  # ensures graph exists up to the bottleneck
    with torch.enable_grad():
        logits = model(x1, x2)
        m = torch.from_numpy(mask.astype(np.float32))
        m = F.pad(m, (0, logits.shape[-1] - w, 0, logits.shape[-2] - h))[None, None]
        score = (logits * m).sum() if m.sum() > 0 else logits.sum()
        act = model.last_diff
        grads = torch.autograd.grad(score, act, retain_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0, :h, :w].detach().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)
