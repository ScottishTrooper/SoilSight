"""SoilSight demo: grain-size estimate from lab tray photos, with range, review flag, evidence and heatmap."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soilsight.predict import Predictor

HERE = Path(__file__).parent
ART = Path(os.environ.get("SOILSIGHT_ARTIFACTS", HERE / "artifacts"))
if not (ART / "config.json").exists():
    from huggingface_hub import snapshot_download
    snapshot_download(os.environ["SOILSIGHT_MODEL_REPO"], local_dir=ART, token=os.environ.get("HF_TOKEN"))
PRED = Predictor(ART, device=os.environ.get("SOILSIGHT_DEVICE", "cpu"))

LOG_DIR = Path("/data/soilsight" if Path("/data").is_dir() else HERE / "log")
(LOG_DIR / "thumbs").mkdir(parents=True, exist_ok=True)
DB = LOG_DIR / "log.db"
COLS = ["id", "time_utc", "sample_id", "n_photos", "pred_gravel", "pred_sand", "pred_fines", "range_gravel",
        "range_sand", "range_fines", "review_needed", "reasons", "decision", "final_gravel", "final_sand",
        "final_fines", "note", "model"]
with sqlite3.connect(DB) as c:
    c.execute(f"CREATE TABLE IF NOT EXISTS log ({', '.join(k + (' INTEGER PRIMARY KEY' if k == 'id' else '') for k in COLS)})")

EXAMPLES = {p.stem: [str(f) for f in sorted(p.glob("*.jpg"))] for p in sorted((HERE / "examples").glob("*")) if p.is_dir()}
FR = ["gravel", "sand", "fines"]


# ------------------------------------------------------------------ helpers
def _load(paths):
    imgs = []
    for p in paths:
        a = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if a is None:
            raise gr.Error(f"Could not read {Path(p).name} as an image.")
        imgs.append(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    return imgs


def _mm(name: str) -> float:
    return float(name.replace("%_", "").replace("mm", ""))


def evidence_plot(r: dict):
    sizes = [_mm(s) for s in r["sieve_names"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=110)
    for x0, x1, lab, col in [(5, 100, "gravel", "#f2e6d0"), (0.08, 5, "sand", "#eef2e0"), (0.001, 0.08, "fines", "#e3ecf4")]:
        ax.axvspan(x0, x1, color=col, zorder=0)
        ax.text(np.sqrt(x0 * x1) if x0 > 0.001 else 0.03, 103, lab, ha="center", va="bottom", fontsize=9, color="#555")
    for n in r["neighbours"]:
        ax.plot(sizes, n["sieves"], lw=1.2, alpha=0.75, label=f"{n['sample_id']} ({n['type']}, sim {n['similarity']:.2f})")
    f, h = r["fractions_pct"], r["half_width_pp"]
    est_x = [5.0, 0.08]
    est_y = [f["sand"] + f["fines"], f["fines"]]
    err = [np.nan_to_num(h["gravel"]), np.nan_to_num(h["fines"])]
    ax.errorbar(est_x, est_y, yerr=err, fmt="o", color="black", ms=7, capsize=5, lw=2, label="estimate (this sample)", zorder=5)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlim(100, 0.03); ax.set_ylim(0, 110)
    ticks = [50, 20, 5, 2, 0.5, 0.16, 0.08]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("particle size (mm)"); ax.set_ylabel("% passing")
    ax.set_title("Estimate vs the 5 most similar lab-tested samples", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def summary_md(r: dict, sample_id: str) -> str:
    f, h = r["fractions_pct"], r["half_width_pp"]
    rng = lambda k: "" if np.isnan(h[k]) else f" ± {h[k]:.0f}"
    status = ("### ⚠️ Review needed\n" + "\n".join(f"- {x}" for x in r["flags"] + r["qc"])) if r["review_needed"] \
        else "### ✅ Auto-accepted\nAll photos passed the checks and the range is narrow."
    rows = "\n".join(f"| **{k.capitalize()}** | {f[k]:.0f}%{rng(k)} |" for k in FR)
    per = ""
    if len(r["per_photo_pct"]) > 1:
        per = "\n\n**Per photo** (gravel / sand / fines): " + "; ".join(
            f"#{i + 1}: {g:.0f}/{s:.0f}/{fi:.0f}" for i, (g, s, fi) in enumerate(r["per_photo_pct"]))
    return (f"## {sample_id or 'Unnamed sample'}\n\n| Fraction | Estimate (% by mass) |\n|---|---|\n{rows}\n\n"
            f"<sub>Range: {r['range_note']}. Gravel > 5 mm, fines < 0.08 mm.</sub>{per}\n\n{status}\n\n"
            "<sub>Visual estimate only. Plasticity and engineering classification still need lab tests.</sub>")


# ------------------------------------------------------------------ callbacks
def analyze(files, example, sample_id, progress=gr.Progress()):
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not paths and example:
        paths, sample_id = EXAMPLES[example], sample_id or example
    if not paths:
        raise gr.Error("Upload one or more photos of the same sample, or pick an example.")
    progress(0.1, desc="Reading photos")
    imgs = _load(paths)
    progress(0.3, desc="Running the model (this takes a little while on CPU)")
    r = PRED.predict(imgs)
    progress(0.9, desc="Building the evidence and heatmap")
    # thumbnails are sent as image arrays, not file paths: Gradio only serves files from its own folders
    gal = []
    for n in r["neighbours"]:
        im = cv2.imread(n["thumb"]) if Path(n["thumb"]).exists() else None
        if im is not None:
            gal.append((cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                        f"{n['sample_id']} · {n['type']} · lab: G{n['fractions'][0]:.0f} S{n['fractions'][1]:.0f} F{n['fractions'][2]:.0f}"))
    hm, _ = PRED.heatmap(r["overview"], "gravel")
    state = {k: v for k, v in r.items() if k != "overview"} | {"sample_id": sample_id, "n_photos": len(imgs)}
    thumb = cv2.resize(r["overview"], (480, int(480 * r["overview"].shape[0] / r["overview"].shape[1])))
    return (summary_md(r, sample_id), evidence_plot(r), gal, hm, r["overview"], state, thumb,
            *[r["fractions_pct"][k] for k in FR])


def redraw_heatmap(overview, fraction):
    if overview is None:
        return None
    return PRED.heatmap(np.asarray(overview), fraction)[0]


def save(state, thumb, decision, g, s, f, note):
    if not state:
        raise gr.Error("Analyse a sample first.")
    final = [g, s, f] if decision == "Corrected" else [state["fractions_pct"][k] for k in FR]
    if decision == "Corrected" and abs(sum(final) - 100) > 1.5:
        raise gr.Error(f"Corrected fractions must add up to 100% (they add up to {sum(final):.1f}%).")
    p, h = state["fractions_pct"], state["half_width_pp"]
    row = [None, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), state["sample_id"], state["n_photos"],
           p["gravel"], p["sand"], p["fines"], h["gravel"], h["sand"], h["fines"], int(state["review_needed"]),
           "; ".join(state["flags"] + state["qc"]), decision.lower(), *final, note, state["model"]]
    with sqlite3.connect(DB) as c:
        cur = c.execute(f"INSERT INTO log VALUES ({','.join('?' * len(COLS))})", row)
        rid = cur.lastrowid
    if thumb is not None:
        cv2.imwrite(str(LOG_DIR / "thumbs" / f"{rid}.jpg"), cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2BGR))
    return f"Saved record #{rid} ({decision.lower()}).", load_log()


def load_log(query: str = ""):
    with sqlite3.connect(DB) as c:
        df = pd.read_sql("SELECT * FROM log ORDER BY id DESC", c)
    if query:
        df = df[df.apply(lambda r: query.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    return df


def export_csv():
    p = LOG_DIR / "soilsight_log.csv"
    load_log().to_csv(p, index=False)
    return str(p)


# ------------------------------------------------------------------ UI
with gr.Blocks(title="SoilSight") as demo:
    gr.Markdown("# SoilSight\nUpload photos of one soil sample on a lab tray. You get a gravel / sand / fines estimate with a "
                "range, a **review flag** when it is unsure, the **most similar lab-tested samples** as evidence, and a "
                "**grain-size heatmap**. Trained on Photogranulometry v1.0 (Plante St-Cyr et al., 2025; CC BY 4.0).")
    state, ov_state, thumb_state = gr.State(), gr.State(), gr.State()
    with gr.Tab("Analyse"):
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Tray photos of ONE sample (1 or more)", file_count="multiple", file_types=["image"])
                example = gr.Dropdown(choices=list(EXAMPLES), value=None, label="…or try an example (held-out test samples)")
                sid = gr.Textbox(label="Sample ID", placeholder="e.g. BH-03 / 4.5 m")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                out_md = gr.Markdown()
        with gr.Row():
            plot = gr.Plot(label="Evidence: similar lab-tested samples")
            gal = gr.Gallery(label="Most similar lab-tested samples", columns=5, height=260)
        with gr.Row():
            with gr.Column():
                frac = gr.Radio(FR, value="gravel", label="Heatmap shows")
                hm = gr.Image(label="Grain-size heatmap: per-patch estimate (dark purple = 0%, green = 50%, yellow = 100%)", interactive=False)
        gr.Markdown("### Technician check")
        with gr.Row():
            decision = gr.Radio(["Confirmed", "Corrected"], value="Confirmed", label="Decision")
            cg, cs, cf = (gr.Number(label=f"{k.capitalize()} %", precision=1) for k in FR)
        note = gr.Textbox(label="Note (optional)")
        save_btn = gr.Button("Save to log")
        saved = gr.Markdown()
    with gr.Tab("Log"):
        q = gr.Textbox(label="Search the log")
        table = gr.Dataframe(value=load_log, interactive=False, wrap=True)
        exp_btn = gr.Button("Export CSV")
        exp_file = gr.File(label="Download")
    with gr.Tab("About"):
        about = open(HERE / "ABOUT.md").read() if (HERE / "ABOUT.md").exists() else ""
        if (ART / "cv_results.csv").exists():
            cv = pd.read_csv(ART / "cv_results.csv")
            cv = cv[["features", "mae_gravel", "mae_sand", "mae_fines", "mae_mean", "mae_mean_natural"]].round(1)
            about += ("\n\n## Results (5-fold cross-validation, mean absolute error in percentage points)\n\n"
                      + cv.to_markdown(index=False))
        cal = PRED.cfg.get("calib_report", {}).get("conformal")
        if cal:
            about += (f"\n\n**Held-out calibration samples (n={cal['n']}):** {cal['share_auto_accepted']:.0%} auto-accepted; "
                      f"all three fractions inside the range for {cal['joint_coverage']:.0%} of samples.")
        gr.Markdown(about)

    go.click(analyze, [files, example, sid], [out_md, plot, gal, hm, ov_state, state, thumb_state, cg, cs, cf])
    frac.change(redraw_heatmap, [ov_state, frac], hm)
    save_btn.click(save, [state, thumb_state, decision, cg, cs, cf, note], [saved, table])
    q.change(load_log, q, table)
    exp_btn.click(export_csv, None, exp_file)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
