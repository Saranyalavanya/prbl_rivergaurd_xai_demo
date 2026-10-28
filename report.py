from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from .config import REPORT_DIR


def build_pdf(row, impact, regions, alerts_status="N/A", out=None):
    out = out or str(REPORT_DIR / f"report_{row['id']}.pdf")
    s = getSampleStyleSheet()
    doc = SimpleDocTemplate(out, pagesize=A4, title="RiverGuard XAI Report")
    el = [Paragraph("RiverGuard XAI - Sand Mining Detection Report", s["Title"]),
          Paragraph(f"Area of interest: <b>{row['aoi_name']}</b> | Generated: {row['created']}", s["Normal"]),
          Paragraph(f"Period: {row['date_before']} -> {row['date_after']} | Source: {row['source']}", s["Normal"]),
          Spacer(1, 10),
          Paragraph(f"Risk: <b>{row['risk_level']}</b> ({row['risk_score']:.0f}/100) - changed area "
                    f"<b>{row['changed_area_ha']:.2f} ha</b> - alert/field status: {alerts_status}", s["Heading3"])]
    if row.get("image_path"):
        el += [Image(row["image_path"], width=500, height=135), Spacer(1, 8)]
    el.append(Paragraph("Environmental impact assessment", s["Heading3"]))
    t = [[k.replace("_", " "), str(v)] for k, v in impact.items() if k != "notes"]
    tb = Table(t, colWidths=[300, 150]); tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey)]))
    el += [tb, Spacer(1, 6)] + [Paragraph("- " + n, s["Normal"]) for n in impact.get("notes", [])]
    el += [Spacer(1, 10), Paragraph("Detected regions", s["Heading3"])]
    rows = [["#", "Area (ha)", "Dist. river (m)", "Conf.", "Risk", "Level"]] + [
        [r["id"], r["area_ha"], r["dist_to_river_m"], r["confidence"], r["risk_score"], r["risk_level"]] for r in regions]
    tb = Table(rows); tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey),
                                              ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    el += [tb, Spacer(1, 10), Paragraph("Explainability: the Grad-CAM panel highlights the image areas that "
           "drove the deep-learning decision. Detections are indicative and require field verification.", s["Italic"])]
    doc.build(el)
    return out
