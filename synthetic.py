"""Synthetic multi-temporal river scenes (4 bands: B,G,R,NIR; reflectance 0-1).

Used for (a) offline Demo mode and (b) training the change-detection model when no
labelled real dataset is available. Each scene = river + sand bars + farmland/forest.
The 'after' image may contain illegal-mining-like changes (bare sand excavations and
turbid water-filled pits on/near the river) plus harmless distractors (crop-cycle
changes, clouds, seasonal brightness shifts) that are labelled 'no change'.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt

# approximate reflectance signatures [B, G, R, NIR]
SIG = {
    "water":  [0.06, 0.07, 0.05, 0.02],
    "turbid": [0.10, 0.13, 0.14, 0.06],
    "forest": [0.03, 0.06, 0.03, 0.40],
    "crop":   [0.05, 0.09, 0.06, 0.34],
    "soil":   [0.12, 0.16, 0.22, 0.28],
    "sand":   [0.27, 0.34, 0.42, 0.46],
}


def _noise(rng, shape, sigma):
    n = gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-8)


def _paint(img, mask, name, rng, jitter=0.02):
    sig = np.array(SIG[name], dtype=np.float32)[:, None, None]
    tex = 1 + 0.08 * _noise(rng, mask.shape, 1.5)
    img[:, mask] = (sig * tex)[:, mask] + rng.normal(0, jitter, (4, mask.sum()))


def _blob(rng, shape, cx, cy, r):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - cx, yy - cy) / r
    d = d * (1 + 0.35 * _noise(rng, shape, r / 3))
    return d < 1.0


def make_scene(size=128, seed=None, mining=True, clouds=None):
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[:H, :W]
    # meandering river (vertical-ish or horizontal-ish)
    amp, per, ph = rng.uniform(.08, .2) * size, rng.uniform(.8, 1.6), rng.uniform(0, 6.28)
    center = size * rng.uniform(.4, .6) + amp * np.sin(per * 2 * np.pi * yy / size + ph)
    half = rng.uniform(.05, .09) * size
    dist = np.abs(xx - center)
    river = dist < half
    bank = (dist >= half) & (dist < half + rng.uniform(.03, .07) * size)
    sandbar = bank & (_noise(rng, (H, W), 4) > -0.2)
    land_n = _noise(rng, (H, W), 10)
    forest = (land_n > 0.2) & ~river & ~sandbar
    crop = (~forest) & ~river & ~sandbar
    soil = crop & (_noise(rng, (H, W), 6) > 0.5)
    crop &= ~soil

    before = np.zeros((4, H, W), np.float32)
    for name, m in (("forest", forest), ("crop", crop), ("soil", soil),
                    ("sand", sandbar), ("water", river)):
        _paint(before, m, name, rng)
    after = before.copy()

    # seasonal shift (global, harmless)
    season = rng.uniform(-0.05, 0.05)
    after[3] *= 1 + season * 2
    after[:3] *= 1 + season

    truth = np.zeros((H, W), bool)

    # harmless distractor: rectangular crop-cycle change far from river
    for _ in range(rng.integers(0, 3)):
        x0, y0 = rng.integers(0, W - 20), rng.integers(0, H - 20)
        w, h = rng.integers(10, 26), rng.integers(8, 22)
        m = np.zeros((H, W), bool); m[y0:y0 + h, x0:x0 + w] = True
        m &= ~river & ~bank
        _paint(after, m, rng.choice(["soil", "crop"]), rng)

    if mining:
        for _ in range(rng.integers(1, 4)):
            side = rng.choice([-1, 1])
            y = rng.integers(10, H - 10)
            cx = center[int(y), 0] + side * rng.uniform(0, half * 2.0)
            r = rng.uniform(.05, .11) * size
            blob = _blob(rng, (H, W), cx, y, r)
            kind = rng.choice(["sand", "turbid", "both"])
            if kind in ("sand", "both"):
                _paint(after, blob, "sand", rng)
            if kind in ("turbid", "both"):
                pit = _blob(rng, (H, W), cx + side * r * .3, y, r * .55)
                _paint(after, pit, "turbid", rng)
            truth |= blob
        # channel widening where mining occurs
        truth &= True

    # clouds in AFTER image (no-change distractor)
    if clouds is None:
        clouds = rng.random() < 0.25
    if clouds:
        cm = gaussian_filter(_noise(rng, (H, W), 8), 2) > 1.1
        cm = gaussian_filter(cm.astype(float), 2)
        for b in range(4):
            after[b] = after[b] * (1 - cm) + 0.85 * cm
        truth &= cm < 0.3

    before = np.clip(before + rng.normal(0, .005, before.shape), 0, 1).astype(np.float32)
    after = np.clip(after + rng.normal(0, .005, after.shape), 0, 1).astype(np.float32)
    return before, after, truth.astype(np.float32)


def make_demo_pair(size=256, seed=7):
    """Larger deterministic scene for the app's Demo mode."""
    b, a, t = make_scene(size, seed=seed, mining=True, clouds=False)
    return b, a, t
