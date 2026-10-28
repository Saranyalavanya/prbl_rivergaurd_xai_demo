"""Image quality / cloud-cover analysis and preprocessing."""
import numpy as np
from skimage.exposure import match_histograms
from scipy.ndimage import binary_dilation, uniform_filter


def cloud_mask(img, scl=None):
    """Boolean cloud mask. Uses Sentinel-2 SCL if provided, else a spectral heuristic:
    clouds are bright AND spectrally flat (white); sand is bright but R >> B."""
    if scl is not None:
        m = np.isin(scl, [3, 8, 9, 10])          # shadow, cloud med/high, cirrus
    else:
        B, G, R, N = img
        bright = (B > 0.30) & (G > 0.30) & (R > 0.30)
        flat = (np.abs(B - R) < 0.10) & (np.abs(G - B) < 0.08)
        m = bright & flat
    m = uniform_filter(m.astype(float), 3) > 0.5
    return binary_dilation(m, iterations=2)


def quality_report(img, scl=None, max_cloud_pct=20.0):
    cm = cloud_mask(img, scl)
    nodata = float(np.mean(~np.isfinite(img).all(0)))
    clear = img[:, ~cm & np.isfinite(img).all(0)] if (~cm).any() else img.reshape(4, -1)
    contrast = float(np.mean(np.std(clear, axis=1))) if clear.size else 0.0
    cloud_pct = float(cm.mean() * 100)
    ok = cloud_pct <= max_cloud_pct and nodata < 0.05 and contrast > 0.01
    reason = "OK"
    if cloud_pct > max_cloud_pct: reason = f"Cloud cover {cloud_pct:.1f}% > {max_cloud_pct}%"
    elif nodata >= 0.05: reason = f"No-data {nodata*100:.1f}%"
    elif contrast <= 0.01: reason = "Low contrast / blank image"
    return {"ok": ok, "cloud_pct": cloud_pct, "nodata_pct": nodata * 100,
            "contrast": contrast, "reason": reason, "cloud_mask": cm}


def indices(img):
    B, G, R, N = img
    e = 1e-6
    return {"ndwi": (G - N) / (G + N + e),         # water
            "ndvi": (N - R) / (N + R + e),         # vegetation
            "sand": (R - B) / (R + B + e)}         # bare sand / soil proxy


def preprocess_pair(before, after):
    """NaN fill, clip, radiometric normalisation (histogram-match 'after' to 'before')."""
    out = []
    for im in (before, after):
        im = np.nan_to_num(im, nan=0.0).astype(np.float32)
        out.append(np.clip(im, 0, 1))
    b, a = out
    a = match_histograms(a, b, channel_axis=0).astype(np.float32)
    return b, a


def to_rgb(img, gain=3.0):
    rgb = np.stack([img[2], img[1], img[0]], -1) * gain
    return np.clip(rgb, 0, 1) ** (1 / 1.2)
