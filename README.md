# 🌊 RiverGuard XAI
**Explainable AI for Illegal Sand Mining Detection Using Multi-Temporal Satellite Imagery**

A complete implementation of the project flowchart:

| Flowchart block | Where it is implemented |
|---|---|
| Create Account / Login | `app.py` (auth screen), `riverguard/db.py` (PBKDF2-hashed passwords) |
| Define AOI, Select Time Period | `app.py` → *New Analysis* page |
| Collect Sentinel-2 images | `riverguard/satellite.py` (Microsoft Planetary Computer STAC, free, no key) |
| Image Quality & Cloud-cover Analysis → OK? → Discard | `riverguard/preprocess.py::quality_report`, `satellite.fetch_period` |
| Preprocess images, multi-temporal comparison | `riverguard/preprocess.py` (clip, histogram matching, indices) |
| Detect potential change regions / Deep Learning model | `riverguard/model.py` (Siamese U-Net), `train.py` |
| Explainable AI (Grad-CAM) | `riverguard/xai.py` |
| Risk assessment / Visual evidence / Environmental impact | `riverguard/analysis.py` |
| Generate alert / Store results in DB | `analysis.run_analysis`, `riverguard/db.py` (SQLite) |
| Admin: dashboard, manage users, set alerts, alerts & notifications, reports, field verification | `app.py` admin pages, `riverguard/report.py` |

## 1. Install
Requires **Python 3.9 – 3.12**.
```bash
cd RiverGuardXAI
python -m venv venv
# Windows:  venv\Scripts\activate        macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```
> `rasterio` is only needed for *Live* mode. If it fails to install on Windows, you can still
> run everything in **Demo mode**: remove the last three lines of `requirements.txt` first
> (`pystac-client`, `planetary-computer`, `rasterio`), or use `conda install -c conda-forge rasterio`.
> For a smaller CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## 2. Train the model (optional – a trained model is already included)
```bash
python train.py --epochs 12 --samples 160     # ~4 min on CPU, writes models/riverguard.pt
```
If `models/riverguard.pt` is missing, the app trains it automatically on first run.

## 3. Test from the command line
```bash
python run_demo_cli.py
```

## 4. Start the web app
```bash
streamlit run app.py
```
Open http://localhost:8501

* **Admin login:** `admin` / `Admin@123` (change it after first login via *Manage Users* → create your own admin).
* **Normal user:** click *Create Account*, then log in.

## 5. Using it
1. **New Analysis** → choose **Demo** (offline, synthetic river scene with simulated mining) or **Live Sentinel-2**.
2. Set the AOI (bounding box lon/lat – a preset for the Cauvery near Karur is included) and the *Before* / *After* date ranges.
3. Click **Run detection**. The app shows quality/cloud checks (bad images are *discarded*), change regions,
   the Grad-CAM heat-map (XAI), a per-region risk score (area, distance to river, confidence, XAI focus, sand gain),
   and the environmental-impact assessment.
4. If risk ≥ the admin threshold, an **alert** is created.
5. **Admin:** *Dashboard*, *Alerts & Field Verification* (set Pending / Under Field Verification / Confirmed / False Positive + notes),
   *Set Alerts* (thresholds), *Manage Users*, *Reports* (PDF per analysis, CSV of all results).

Tip for Live mode: use a dry-season pair of periods (e.g. Before = Feb–Mar 2025, After = Feb–Mar 2026),
a 5–10 km AOI, and the same season for both to avoid false changes from monsoon/crop cycles.

## 6. Using real labelled data (recommended for final accuracy)
The bundled model is trained on **procedurally generated** scenes so the project runs anywhere.
For real deployment, fine-tune on labelled Sentinel-2 pairs (e.g. OSCD, LEVIR-CD, or your own
polygons of mined sites): edit `SceneDataset` in `train.py` to return `(before[4,H,W], after[4,H,W], mask[H,W])`
and run `python train.py`. Always confirm detections with **field verification**.

## Project layout
```
app.py                Streamlit UI (user + admin)      train.py          model training
run_demo_cli.py       headless test                    requirements.txt
riverguard/ config.py db.py satellite.py preprocess.py synthetic.py
            model.py xai.py analysis.py report.py
models/ (weights)   data/ (SQLite DB)   reports/ (figures, PDFs)
```
