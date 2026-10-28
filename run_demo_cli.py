"""Headless end-to-end test (no web UI):  python run_demo_cli.py"""
from riverguard import analysis, satellite, db
from riverguard.model import load_model
from riverguard.config import MODEL_PATH
from riverguard.preprocess import quality_report, preprocess_pair

db.init_db()
model = load_model(MODEL_PATH)
imgs = satellite.demo_images(None, None)
print("Quality T1:", quality_report(imgs["before"])["reason"], "| T2:", quality_report(imgs["after"])["reason"])
b, a = preprocess_pair(imgs["before"], imgs["after"])
res = analysis.run_analysis(model, b, a, db.get_settings(), tag="cli_demo")
t, p = imgs["truth"] > .5, res["mask"]
print(f"Risk: {res['risk_level']} ({res['risk_score']}/100)  area: {res['area_ha']:.2f} ha  regions: {len(res['regions'])}")
print(f"Dice vs synthetic truth: {2*(t&p).sum()/max(t.sum()+p.sum(),1):.2f}")
print("Alert:", res["alert_message"] or "none")
print("Evidence figure:", res["evidence_path"])
for n in res["impact"]["notes"]: print(" -", n)
