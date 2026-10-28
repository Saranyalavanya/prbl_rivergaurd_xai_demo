from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "riverguard.db"
MODEL_PATH = ROOT / "models" / "riverguard.pt"
REPORT_DIR = ROOT / "reports"
for p in (DB_PATH.parent, MODEL_PATH.parent, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_M = 10.0           # Sentinel-2 10 m bands (B02,B03,B04,B08)
BANDS = ["B02", "B03", "B04", "B08"]   # Blue, Green, Red, NIR
MAX_CLOUD_PCT = 20.0          # quality gate: discard images above this
DEFAULT_SETTINGS = {
    "alert_min_risk": 40,         # risk score (0-100) that triggers an alert
    "alert_min_area_ha": 0.5,     # minimum changed area (hectares)
    "change_threshold": 0.5,      # model probability threshold
}
DEFAULT_ADMIN = ("admin", "Admin@123")
