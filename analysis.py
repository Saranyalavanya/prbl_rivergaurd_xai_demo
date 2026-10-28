import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
"""Core pipeline: change detection -> XAI -> risk -> environmental impact -> alert."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, binary_opening, disk
from .config import PIXEL_SIZE_M, REPORT_DIR
from .model import predict_prob
from .xai import grad_cam
from .preprocess import indices, to_rgb, cloud_mask

HA_PER_PX = (PIXEL_SIZE_M ** 2) / 10_000.0
def set_pixel_size(m):
    """Set ground resolution (metres/pixel) used for area & distance calculations."""
    global PIXEL_SIZE_M, HA_PER_PX
    PIXEL_SIZE_M = float(m)
    HA_PER_PX = PIXEL_SIZE_M ** 2 / 10_000.0


LEVELS = [(25, "Low"), (50, "Medium"), (75, "High"), (101, "Critical")]


def level_of(score):
    return next(n for t, n in LEVELS if score < t)


def river_mask(img):
    """Largest connected water body in the 'before' image."""
    w = indices(img)["ndwi"] > 0.1
    lab, n = ndi.label(w)
    if n == 0:
        return np.zeros_like(w)
    sizes = ndi.sum(w, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def detect_changes(model, before, after, threshold=0.5, min_px=20):
    prob = predict_prob(model, before, after)
    mask = prob > threshold
    mask = binary_opening(mask, disk(1))
    mask = remove_small_objects(mask, min_px)
    return prob, mask


def assess_regions(prob, mask, cam, before, after, river):
    """Risk assessment per detected region (0-100)."""
    lab, n = ndi.label(mask)
    dist = ndi.distance_transform_edt(~river) * PIXEL_SIZE_M if river.any() else np.full(mask.shape, 1e4)
    ib, ia = indices(before), indices(after)
    cam_total = cam.sum() + 1e-8
    regions = []
    for i in range(1, n + 1):
        m = lab == i
        area = m.sum() * HA_PER_PX
        d = float(dist[m].min())
        conf = float(prob[m].mean())
        focus = float(cam[m].sum() / cam_total)            # share of XAI attention
        sand_gain = float(np.clip((ia["sand"][m] - ib["sand"][m]).mean() / 0.3, 0, 1))
        score = (min(area / 5.0, 1) * 30 + np.exp(-d / 150.0) * 30 +
                 conf * 15 + min(focus * 3, 1) * 10 + sand_gain * 15)
        ys, xs = np.nonzero(m)
        regions.append({"id": i, "area_ha": round(area, 3), "dist_to_river_m": round(d, 1),
                        "confidence": round(conf, 3), "xai_focus": round(focus, 3),
                        "sand_gain": round(sand_gain, 3), "risk_score": round(float(score), 1),
                        "risk_level": level_of(score),
                        "centroid_xy": (int(xs.mean()), int(ys.mean())),
                        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})
    regions.sort(key=lambda r: -r["risk_score"])
    overall = max([r["risk_score"] for r in regions], default=0.0)
    if len(regions) > 1:   # several hotspots raise overall concern slightly
        overall = min(100.0, overall + 3 * (len(regions) - 1))
    return regions, round(float(overall), 1)


def environmental_impact(mask, before, after, river):
    ib, ia = indices(before), indices(after)
    area = mask.sum() * HA_PER_PX
    near = ndi.binary_dilation(river, iterations=3) if river.any() else np.zeros_like(mask)
    water_b, water_a = (ib["ndwi"] > 0.1).sum(), (ia["ndwi"] > 0.1).sum()
    veg_drop = float(np.clip((ib["ndvi"][mask] - ia["ndvi"][mask]).mean(), 0, 2)) if mask.any() else 0.0
    wmask = (ia["ndwi"] > 0.1) & ndi.binary_dilation(mask, iterations=15)
    turb = 0.0
    if wmask.any():
        rg_a = (after[2][wmask] / (after[1][wmask] + 1e-6)).mean()
        rg_b = (before[2][wmask] / (before[1][wmask] + 1e-6)).mean()
        turb = float(max(0.0, rg_a - rg_b))
    impact = {
        "disturbed_area_ha": round(float(area), 2),
        "riverbank_or_channel_overlap_ha": round(float((mask & near).sum() * HA_PER_PX), 2),
        "vegetation_loss_ndvi_drop": round(veg_drop, 3),
        "water_area_change_pct": round(100.0 * (water_a - water_b) / max(water_b, 1), 1),
        "turbidity_index_increase": round(turb, 3),
    }
    notes = []
    if impact["riverbank_or_channel_overlap_ha"] > 0.3: notes.append("Excavation touches the river channel/banks: bank erosion and bed lowering risk.")
    if impact["turbidity_index_increase"] > 0.1: notes.append("Increased turbidity near disturbed zones: aquatic habitat and water-quality impact.")
    if impact["vegetation_loss_ndvi_drop"] > 0.2: notes.append("Riparian vegetation loss detected.")
    if abs(impact["water_area_change_pct"]) > 5: notes.append("Noticeable change in water-surface extent (channel widening/pits).")
    if not notes: notes.append("No major secondary environmental impact indicators detected.")
    impact["notes"] = notes
    return impact


def make_evidence_figure(before, after, prob, mask, cam, regions, path):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    ax[0].imshow(to_rgb(before)); ax[0].set_title("Before (T1)")
    ax[1].imshow(to_rgb(after)); ax[1].set_title("After (T2)")
    ax[2].imshow(to_rgb(after)); ax[2].imshow(np.ma.masked_where(~mask, prob), cmap="autumn", alpha=.75, vmin=0, vmax=1)
    ax[2].contour(mask, [0.5], colors="red", linewidths=1.2); ax[2].set_title("Detected change regions")
    ax[3].imshow(to_rgb(after)); im = ax[3].imshow(cam, cmap="jet", alpha=.5, vmin=0, vmax=1)
    ax[3].set_title("Grad-CAM: where the model looked"); plt.colorbar(im, ax=ax[3], fraction=.046)
    for r in regions[:8]:
        x0, y0, x1, y1 = r["bbox"]
        ax[2].add_patch(plt.Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, fill=False, ec="yellow", lw=1.2))
        ax[2].text(x0, max(y0 - 4, 6), f"#{r['id']} {r['risk_level']}", color="yellow", fontsize=8, weight="bold")
    for a in ax: a.axis("off")
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_analysis(model, before, after, settings, tag="analysis"):
    """Full pipeline on two preprocessed (4,H,W) arrays. Returns dict of results."""
    prob, mask = detect_changes(model, before, after, settings.get("change_threshold", 0.5))
    river = river_mask(before)
    cam = grad_cam(model, before, after, mask) if mask.any() else np.zeros(mask.shape, np.float32)
    regions, overall = assess_regions(prob, mask, cam, before, after, river)
    impact = environmental_impact(mask, before, after, river)
    area = float(mask.sum() * HA_PER_PX)
    level = level_of(overall)
    path = str(REPORT_DIR / f"{tag}.png")
    make_evidence_figure(before, after, prob, mask, cam, regions, path)
    alert = overall >= settings["alert_min_risk"] and area >= settings["alert_min_area_ha"]
    msg = (f"{level} risk ({overall:.0f}/100): possible illegal sand mining - "
           f"{len(regions)} region(s), {area:.2f} ha changed.") if alert else ""
    return {"prob": prob, "mask": mask, "cam": cam, "river": river, "regions": regions,
            "risk_score": overall, "risk_level": level, "area_ha": area, "impact": impact,
            "evidence_path": path, "alert": alert, "alert_message": msg}
