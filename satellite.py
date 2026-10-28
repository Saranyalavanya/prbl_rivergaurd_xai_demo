"""Sentinel-2 L2A acquisition (Microsoft Planetary Computer STAC - free, no API key)
and offline demo data. Returns (4,H,W) reflectance arrays [B02,B03,B04,B08]."""
import numpy as np
from .config import BANDS, MAX_CLOUD_PCT
from .preprocess import quality_report
from .synthetic import make_scene

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def demo_images(period_a, period_b, size=128, seed=7):
    b, a, truth = make_scene(size, seed=seed, mining=True, clouds=False)
    return {"before": b, "after": a, "truth": truth,
            "log": ["Demo mode: synthetic Sentinel-2-like scene generated (no internet needed)."],
            "meta": {"before": "synthetic", "after": "synthetic"}}


def _read_item(item, bbox, shape):
    import planetary_computer as pc
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.enums import Resampling
    item = pc.sign(item)
    arrs = []
    for b in BANDS + ["SCL"]:
        with rasterio.open(item.assets[b].href) as src:
            wb = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*wb, transform=src.transform)
            res = Resampling.nearest if b == "SCL" else Resampling.bilinear
            a = src.read(1, window=win, out_shape=shape, resampling=res, boundless=True)
            arrs.append(a.astype(np.float32))
    scl = arrs.pop().astype(np.uint8)
    img = np.stack(arrs)
    offset = -0.1 if float(item.properties.get("s2:processing_baseline", "04.00")) >= 4.0 else 0.0
    img = np.where(img == 0, np.nan, img * 1e-4 + offset)
    return np.clip(img, 0, 1), scl


def fetch_period(bbox, start, end, shape=(512, 512), log=None):
    """Step: Collect Sentinel-2 -> quality & cloud analysis -> discard bad images -> keep best."""
    from pystac_client import Client
    log = log if log is not None else []
    cat = Client.open(STAC_URL)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                            query={"eo:cloud_cover": {"lt": 60}}).items())
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    log.append(f"{start}..{end}: {len(items)} candidate scenes")
    for it in items[:8]:
        img, scl = _read_item(it, bbox, shape)
        q = quality_report(img, scl, MAX_CLOUD_PCT)
        if q["ok"]:
            log.append(f"  ACCEPTED {it.id} ({it.datetime.date()}), AOI cloud {q['cloud_pct']:.1f}%")
            return img, it.id, str(it.datetime.date())
        log.append(f"  DISCARDED {it.id}: {q['reason']}")
    raise RuntimeError(f"No acceptable (cloud-free) image found for {start}..{end}. Try a wider period.")


def live_images(bbox, period_a, period_b, shape=(512, 512)):
    log = []
    b, id_b, d_b = fetch_period(bbox, *period_a, shape=shape, log=log)
    a, id_a, d_a = fetch_period(bbox, *period_b, shape=shape, log=log)
    return {"before": b, "after": a, "truth": None, "log": log,
            "meta": {"before": f"{id_b} ({d_b})", "after": f"{id_a} ({d_a})"}}


def grid_for_bbox(bbox, target_m=10.0, max_px=768):
    """Pixel grid (h, w) and real metres/pixel for a lon/lat bbox."""
    import math
    w_m = (bbox[2] - bbox[0]) * 111_320 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    h_m = (bbox[3] - bbox[1]) * 110_574
    scale = min(1.0, max_px * target_m / max(w_m, h_m))
    w_px, h_px = max(int(w_m * scale / target_m), 32), max(int(h_m * scale / target_m), 32)
    return (h_px, w_px), target_m / scale
