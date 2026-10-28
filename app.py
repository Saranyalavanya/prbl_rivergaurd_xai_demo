"""RiverGuard XAI - Streamlit web app.  Run:  streamlit run app.py"""
import json, datetime as dt
import numpy as np, pandas as pd, streamlit as st
from riverguard import db, analysis, satellite
from riverguard.config import MODEL_PATH, DEFAULT_ADMIN
from riverguard.model import load_model
from riverguard.preprocess import quality_report, preprocess_pair
from riverguard.report import build_pdf

st.set_page_config(page_title="RiverGuard XAI", page_icon="🌊", layout="wide")
db.init_db()


@st.cache_resource(show_spinner=False)
def get_model():
    if not MODEL_PATH.exists():
        import train
        bar = st.progress(0.0, "First run: training the change-detection model (~3-5 min, one time)...")
        train.train(epochs=12, samples=160, verbose=False,
                    progress=lambda f, f1: bar.progress(f, f"Training... val-F1 {f1:.2f}"))
        bar.empty()
    return load_model(MODEL_PATH)


PRESETS = {
    "Cauvery River near Karur, Tamil Nadu": (78.03, 10.93, 78.10, 10.99),
    "Custom (enter coordinates)": (78.03, 10.93, 78.10, 10.99),
}


# =============================================================== AUTH SCREEN
def auth_screen():
    st.title("🌊 RiverGuard XAI")
    st.caption("Explainable AI for Illegal Sand Mining Detection using Multi-Temporal Satellite Imagery")
    t1, t2 = st.tabs(["Login", "Create Account"])
    with t1:
        u = st.text_input("Username", key="lu"); p = st.text_input("Password", type="password", key="lp")
        if st.button("Login", type="primary"):
            user = db.login(u, p)
            if user:
                st.session_state.user = user; st.rerun()
            else:
                st.error("Invalid credentials or inactive account.")
        st.info(f"Default admin → username: **{DEFAULT_ADMIN[0]}**, password: **{DEFAULT_ADMIN[1]}**")
    with t2:
        nu = st.text_input("Choose username"); em = st.text_input("Email")
        np1 = st.text_input("Choose password", type="password")
        if st.button("Create Account"):
            ok, msg = db.create_user(nu, em, np1)
            (st.success if ok else st.error)(msg)


# ========================================================== NEW ANALYSIS PAGE
def page_analysis():
    st.header("New Analysis")
    user = st.session_state.user
    c1, c2 = st.columns([1, 1])
    with c1:
        mode = st.radio("Data source", ["Demo (offline, synthetic)", "Live Sentinel-2 (internet)"], horizontal=True)
        name = st.text_input("AOI name", "Demo River Reach")
        preset = st.selectbox("Area of Interest (AOI)", list(PRESETS))
        bb = PRESETS[preset]
        a, b = st.columns(2)
        minlon = a.number_input("Min lon", value=bb[0], format="%.4f"); minlat = b.number_input("Min lat", value=bb[1], format="%.4f")
        maxlon = a.number_input("Max lon", value=bb[2], format="%.4f"); maxlat = b.number_input("Max lat", value=bb[3], format="%.4f")
        bbox = [minlon, minlat, maxlon, maxlat]
    with c2:
        st.markdown("**Select time period**")
        today = dt.date.today()
        d1 = st.date_input("Before period (T1)", (today - dt.timedelta(days=455), today - dt.timedelta(days=365)), key="d1")
        d2 = st.date_input("After period (T2)", (today - dt.timedelta(days=60), today), key="d2")
        st.map(pd.DataFrame({"lat": [(minlat + maxlat) / 2], "lon": [(minlon + maxlon) / 2]}), zoom=11, height=220)
    if not (isinstance(d1, tuple) and len(d1) == 2 and isinstance(d2, tuple) and len(d2) == 2):
        st.warning("Pick a start AND end date for both periods."); return

    if not st.button("▶ Run detection", type="primary"):
        return
    model = get_model()
    settings = db.get_settings()
    live = mode.startswith("Live")
    with st.status("Running RiverGuard pipeline...", expanded=True) as status:
        try:
            st.write("1️⃣ Collecting satellite images + quality / cloud-cover analysis")
            if live:
                shape, px_m = satellite.grid_for_bbox(bbox)
                imgs = satellite.live_images(bbox, [str(x) for x in d1], [str(x) for x in d2], shape)
            else:
                px_m = 10.0
                imgs = satellite.demo_images(d1, d2)
            analysis.set_pixel_size(px_m)
            for line in imgs["log"]: st.write("   " + line)
            qb, qa = quality_report(imgs["before"]), quality_report(imgs["after"])
            for lbl, q in (("T1", qb), ("T2", qa)):
                st.write(f"   {lbl}: cloud {q['cloud_pct']:.1f}% → {'✅ OK' if q['ok'] else '❌ ' + q['reason'] + ' (discard)'}")
            if not (qb["ok"] and qa["ok"]):
                status.update(label="Image quality not OK - image discarded. Choose another time period.", state="error"); return
            st.write("2️⃣ Preprocessing + multi-temporal comparison")
            before, after = preprocess_pair(imgs["before"], imgs["after"])
            st.write("3️⃣ Deep-learning change detection, Grad-CAM, risk & impact assessment")
            res = analysis.run_analysis(model, before, after, settings, tag=f"u{user['id']}_{dt.datetime.now():%Y%m%d_%H%M%S}")
            st.write("4️⃣ Storing results in database / generating alert")
            rid = db.save_result(user["id"], name, bbox, str(d1[0]), str(d2[1]), "Live" if live else "Demo",
                                 res["area_ha"], res["risk_score"], res["risk_level"], res["impact"], res["evidence_path"])
            if res["alert"]:
                db.create_alert(rid, res["alert_message"], res["risk_level"])
            status.update(label="Analysis complete", state="complete")
        except Exception as e:
            status.update(label=f"Failed: {e}", state="error"); st.exception(e); return

    k = st.columns(4)
    k[0].metric("Risk level", res["risk_level"], f"{res['risk_score']:.0f}/100")
    k[1].metric("Changed area", f"{res['area_ha']:.2f} ha")
    k[2].metric("Regions", len(res["regions"]))
    k[3].metric("Alert", "RAISED 🚨" if res["alert"] else "None")
    if res["alert"]: st.error(res["alert_message"])
    else: st.success("No alert: below the configured risk/area thresholds.")
    st.subheader("Visual evidence + Explainable AI (Grad-CAM)")
    st.image(res["evidence_path"], width="stretch")
    if imgs.get("truth") is not None:
        t = imgs["truth"] > 0.5; p = res["mask"]
        f1 = 2 * (t & p).sum() / max(t.sum() + p.sum(), 1)
        st.caption(f"Demo ground-truth agreement (F1 / Dice): **{f1:.2f}**")
    st.subheader("Risk assessment per region")
    st.dataframe(pd.DataFrame(res["regions"]).drop(columns=["bbox"], errors="ignore"), width="stretch")
    st.subheader("Environmental impact assessment")
    imp = res["impact"]
    st.dataframe(pd.DataFrame([(k.replace("_", " "), v) for k, v in imp.items() if k != "notes"], columns=["Indicator", "Value"]),
                 width="stretch", hide_index=True)
    for n in imp["notes"]: st.markdown("- " + n)


def page_history():
    st.header("My Analyses" if st.session_state.user["role"] != "admin" else "All Analyses")
    rows = db.list_results(None if st.session_state.user["role"] == "admin" else st.session_state.user["id"])
    if not rows: st.info("No analyses yet."); return
    df = pd.DataFrame(rows).drop(columns=["impact", "bbox"])
    st.dataframe(df, width="stretch")
    sel = st.selectbox("View evidence for result #", [r["id"] for r in rows])
    r = next(x for x in rows if x["id"] == sel)
    st.image(r["image_path"], width="stretch")


# ================================================================= ADMIN PAGES
def page_dashboard():
    st.header("Overall Dashboard")
    res, al, us = db.list_results(), db.list_alerts(), db.list_users()
    c = st.columns(4)
    c[0].metric("Users", len(us)); c[1].metric("Analyses", len(res))
    c[2].metric("Alerts", len(al)); c[3].metric("Pending verification", sum(a["status"] == "Pending" for a in al))
    if res:
        df = pd.DataFrame(res)
        a, b = st.columns(2)
        a.subheader("Risk level distribution")
        a.bar_chart(df["risk_level"].value_counts())
        b.subheader("Changed area per analysis (ha)")
        b.line_chart(df.sort_values("id").set_index("id")["changed_area_ha"])
        st.subheader("Recent analyses")
        st.dataframe(df.drop(columns=["impact", "bbox"]).head(10), width="stretch")
    else:
        st.info("No data yet. Run an analysis first.")


def page_users():
    st.header("Manage Users")
    us = db.list_users()
    st.dataframe(pd.DataFrame(us), width="stretch")
    sel = st.selectbox("User", [f"{u['id']} - {u['username']}" for u in us])
    uid = int(sel.split(" - ")[0]); u = next(x for x in us if x["id"] == uid)
    a, b, c = st.columns(3)
    if a.button("Deactivate" if u["active"] else "Activate"):
        db.update_user(uid, active=0 if u["active"] else 1); st.rerun()
    if b.button("Make user" if u["role"] == "admin" else "Make admin"):
        db.update_user(uid, role="user" if u["role"] == "admin" else "admin"); st.rerun()
    if c.button("Delete user"):
        db.delete_user(uid); st.rerun()


def page_set_alerts():
    st.header("Set Alerts")
    s = db.get_settings()
    risk = st.slider("Alert when risk score ≥", 0, 100, int(s["alert_min_risk"]))
    area = st.number_input("…and changed area ≥ (hectares)", 0.0, 100.0, float(s["alert_min_area_ha"]), 0.1)
    thr = st.slider("Model change-probability threshold", 0.1, 0.9, float(s["change_threshold"]), 0.05)
    if st.button("Save alert settings", type="primary"):
        db.set_setting("alert_min_risk", risk); db.set_setting("alert_min_area_ha", area); db.set_setting("change_threshold", thr)
        st.success("Saved.")


def page_alerts():
    st.header("Alerts & Notifications  +  Field Verification")
    al = db.list_alerts()
    if not al: st.info("No alerts."); return
    for a in al:
        icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(a["level"], "⚪")
        with st.expander(f"{icon} Alert #{a['id']} - {a['aoi_name']} - {a['level']} - [{a['status']}]"):
            st.write(a["message"]); st.caption(f"Raised: {a['created']}")
            st.image(a["image_path"], width="stretch")
            opts = ["Pending", "Under Field Verification", "Confirmed Illegal Mining", "False Positive"]
            stt = st.selectbox("Field verification status", opts, opts.index(a["status"]), key=f"s{a['id']}")
            notes = st.text_area("Field notes", a["field_notes"], key=f"n{a['id']}")
            if st.button("Update", key=f"u{a['id']}"):
                db.update_alert(a["id"], stt, notes); st.rerun()


def page_reports():
    st.header("Generate Reports")
    rows = db.list_results()
    if not rows: st.info("No analyses yet."); return
    df = pd.DataFrame(rows).drop(columns=["impact", "bbox"])
    st.download_button("⬇ Export all results (CSV)", df.to_csv(index=False), "riverguard_results.csv", "text/csv")
    sel = st.selectbox("PDF report for result #", [r["id"] for r in rows])
    r = next(x for x in rows if x["id"] == sel)
    if st.button("Generate PDF"):
        al = [a for a in db.list_alerts() if a["result_id"] == sel]
        path = build_pdf(r, json.loads(r["impact"]), [], al[0]["status"] if al else "No alert")
        with open(path, "rb") as f:
            st.download_button("⬇ Download PDF", f.read(), f"riverguard_report_{sel}.pdf", "application/pdf")


# ====================================================================== ROUTER
if "user" not in st.session_state:
    auth_screen(); st.stop()
user = st.session_state.user
admin = user["role"] == "admin"
pages = {"New Analysis": page_analysis, "History": page_history}
if admin:
    pages.update({"Dashboard": page_dashboard, "Alerts & Field Verification": page_alerts,
                  "Set Alerts": page_set_alerts, "Manage Users": page_users, "Reports": page_reports})
with st.sidebar:
    st.title("🌊 RiverGuard XAI")
    st.write(f"👤 **{user['username']}** ({user['role']})")
    if admin:
        pend = sum(a["status"] == "Pending" for a in db.list_alerts())
        if pend: st.warning(f"🔔 {pend} alert(s) pending verification")
    choice = st.radio("Navigate", list(pages))
    if st.button("Logout"):
        del st.session_state["user"]; st.rerun()
pages[choice]()
