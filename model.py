"""Siamese U-Net-style change-detection network (PyTorch)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class SiameseChangeNet(nn.Module):
    def __init__(self, in_ch=4, c=(16, 32, 64, 64)):
        super().__init__()
        self.e1, self.e2 = block(in_ch, c[0]), block(c[0], c[1])
        self.e3, self.e4 = block(c[1], c[2]), block(c[2], c[3])
        # decoder: each level sees [|f1-f2|, f1, f2]
        self.d3 = block(c[3] * 3 + c[2] * 3, c[2])
        self.d2 = block(c[2] + c[1] * 3, c[1])
        self.d1 = block(c[1] + c[0] * 3, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)
        self.last_diff = None   # bottleneck activation (Grad-CAM target layer)

    def encode(self, x):
        f1 = self.e1(x)
        f2 = self.e2(F.max_pool2d(f1, 2))
        f3 = self.e3(F.max_pool2d(f2, 2))
        f4 = self.e4(F.max_pool2d(f3, 2))
        return f1, f2, f3, f4

    @staticmethod
    def _cat(a, b):
        return torch.cat([(a - b).abs(), a, b], 1)

    def forward(self, x1, x2):
        a, b = self.encode(x1), self.encode(x2)
        s = [self._cat(a[i], b[i]) for i in range(4)]
        diff4 = s[3]
        self.last_diff = diff4
        y = F.interpolate(diff4, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.d3(torch.cat([y, s[2]], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.d2(torch.cat([y, s[1]], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.d1(torch.cat([y, s[0]], 1))
        return self.out(y)


# ------------------------------------------------------------------ helpers
MEAN, STD = 0.18, 0.15


def to_tensor(img):
    return ((torch.from_numpy(img).float() - MEAN) / STD).unsqueeze(0)


def _pad8(x):
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), h, w


@torch.no_grad()
def predict_prob(model, before, after):
    model.eval()
    x1, h, w = _pad8(to_tensor(before)); x2, _, _ = _pad8(to_tensor(after))
    return torch.sigmoid(model(x1, x2))[0, 0, :h, :w].numpy()


def load_model(path):
    m = SiameseChangeNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
