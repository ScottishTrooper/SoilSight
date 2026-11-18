"use strict";
const FR = ["gravel", "sand", "fines"];
const COL = { gravel: "var(--gravel)", sand: "var(--sand)", fines: "var(--fines)" };
const NB_COLORS = ["#7b5ea7", "#3f8f8a", "#c0623b", "#4f79b5", "#9a9a3a"];
const $ = (s) => document.querySelector(s);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => (v == null || Number.isNaN(v) ? "–" : `${Number(v).toFixed(d)}`);

let DATA = null, filter = "all", sortKey = "id", current = null, heat = "none";

fetch("data.json").then((r) => r.json()).then((d) => { DATA = d; init(); })
  .catch((e) => { $("#detail").innerHTML = `<p>Could not load data.json (${esc(e)}).</p>`; });

function meanErr(s) { return FR.reduce((a, f) => a + Math.abs(s.est[f] - s.lab[f]), 0) / 3; }

function init() {
  const t = DATA.summary.test, n = DATA.samples.length;
  $("#n-test").textContent = n;
  const stats = [
    [`${pct(t.mae_mean, 1)} pp`, `average error on ${n} held-out samples`],
    [`${pct(100 * t.share_auto_accepted)}%`, `auto-accepted by the review flag`],
    [`${pct(t.mae_mean_accepted, 1)} pp`, `average error on auto-accepted samples`],
    [`${pct(DATA.summary.baseline_mae, 1)} pp`, `error of always guessing the average`],
  ];
  $("#stats").innerHTML = stats.map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  renderCV();
  $("#model-version").textContent = DATA.summary.model ? `Model: ${DATA.summary.model}` : "";
  document.querySelectorAll("#filter button").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.f;
    document.querySelectorAll("#filter button").forEach((x) => x.classList.toggle("on", x === b));
    renderList();
  }));
  $("#sort").addEventListener("change", (e) => { sortKey = e.target.value; renderList(); });
  renderList();
  const first = visible()[0];
  if (first) select(first.id);
}

function visible() {
  let xs = DATA.samples.filter((s) => filter === "all" || s.status === filter);
  if (sortKey === "fines") xs = xs.slice().sort((a, b) => a.lab.fines - b.lab.fines);
  else if (sortKey === "err") xs = xs.slice().sort((a, b) => meanErr(b) - meanErr(a));
  else xs = xs.slice().sort((a, b) => a.id.localeCompare(b.id));
  return xs;
}

function renderList() {
  const xs = visible();
  $("#list").innerHTML = xs.map((s) => `
    <div class="item ${s.id === current ? "sel" : ""}" data-id="${esc(s.id)}">
      <img src="${esc(s.thumb)}" alt="" loading="lazy">
      <div><div class="t">${esc(s.id)} <span class="chip ${s.status === "accepted" ? "ok" : "warn"}">${s.status === "accepted" ? "accepted" : "review"}</span></div>
      <div class="s">${esc(s.type)} · est. G${pct(s.est.gravel)} S${pct(s.est.sand)} F${pct(s.est.fines)}<br>lab G${pct(s.lab.gravel)} S${pct(s.lab.sand)} F${pct(s.lab.fines)}</div></div>
    </div>`).join("") || `<p class="muted" style="padding:10px">No samples.</p>`;
  document.querySelectorAll(".item").forEach((el) => el.addEventListener("click", () => select(el.dataset.id)));
}

function select(id) {
  current = id; heat = "none";
  document.querySelectorAll(".item").forEach((el) => el.classList.toggle("sel", el.dataset.id === id));
  renderDetail(DATA.samples.find((s) => s.id === id));
}

function stackBar(v) {
  return `<div class="bar">${FR.map((f) => `<div title="${f} ${pct(v[f])}%" style="width:${Math.max(0, v[f])}%;background:${COL[f]}"></div>`).join("")}</div>`;
}

function renderDetail(s) {
  const ok = s.status === "accepted";
  const rows = FR.map((f) => {
    const h = s.half[f], e = Math.abs(s.est[f] - s.lab[f]), inside = e <= h;
    return `<tr><td><span class="sw" style="background:${COL[f]}"></span>${f[0].toUpperCase() + f.slice(1)}</td>
      <td><b>${pct(s.est[f])}%</b> <span class="muted">± ${pct(h)}</span></td><td>${pct(s.lab[f])}%</td>
      <td class="${inside ? "in" : "out"}">${pct(e, 1)} ${inside ? "✓" : "✗"}</td></tr>`;
  }).join("");
  const reasons = s.reasons.length ? `<ul>${s.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "";
  let perPhoto = "";
  if (s.per_photo.length > 4) {
    const rng = (j) => { const v = s.per_photo.map((p) => p[j]); return `${pct(Math.min(...v))}–${pct(Math.max(...v))}%`; };
    perPhoto = `<div class="note">Across the ${s.per_photo.length} photos: gravel ${rng(0)}, sand ${rng(1)}, fines ${rng(2)}.</div>`;
  } else if (s.per_photo.length > 1) {
    perPhoto = `<div class="note">Per photo (G/S/F): ${s.per_photo.map((p, i) => `#${i + 1} ${pct(p[0])}/${pct(p[1])}/${pct(p[2])}`).join(" · ")}</div>`;
  }
  $("#detail").innerHTML = `
    <div class="dhead"><h2>Sample ${esc(s.id)}</h2><span class="muted">${esc(s.type)} · ${s.n_photos} photo${s.n_photos > 1 ? "s" : ""} · held-out test sample</span></div>
    <div class="status ${ok ? "ok" : "warn"}"><b>${ok ? "✅ Auto-accepted" : "⚠️ Review needed"}</b>
      ${ok ? " – the range is narrow and similar lab samples agree." : reasons}</div>
    <div class="grid2">
      <div>
        <div class="photo"><img id="photo" src="${esc(s.img)}" alt="Tray photo of sample ${esc(s.id)}"></div>
        <div class="hm-ctl">Heatmap:
          ${["none", ...FR].map((f) => `<button data-h="${f}" class="${f === "none" ? "on" : ""}">${f === "none" ? "off" : f}</button>`).join("")}
        </div>
        <div id="hm-scale" style="display:none"><div class="scale"></div><div class="scale-l"><span>0%</span><span>50%</span><span>100%</span></div>
          <div class="note">Per-patch estimate of the selected fraction (each cell ≈ 4 mm of tray).</div></div>
      </div>
      <div>
        <table class="fr"><tr><th>Fraction</th><th>Estimate ± 90% range</th><th>Lab result</th><th>Error (pp)</th></tr>${rows}</table>
        ${stackBar(s.est)}<div class="bar-l"><span>estimate</span></div>
        ${stackBar(s.lab)}<div class="bar-l"><span>lab sieve result</span></div>
        <div class="note">✓ = lab value inside the stated range. ${esc(s.range_note.charAt(0).toUpperCase() + s.range_note.slice(1))}.</div>${perPhoto}
      </div>
    </div>
    <h3>Evidence: the 5 most similar lab-tested samples</h3>
    <div class="plot">${plotSVG(s)}</div>
    <div class="legend">${s.neighbours.map((n, i) => `<span><i style="background:${NB_COLORS[i]}"></i>${esc(n.sample_id)} (${esc(n.type)}, similarity ${n.similarity.toFixed(2)})</span>`).join("")}
      <span><i style="background:#1f1d1a"></i>this sample's lab curve</span><span>● estimate ± range</span></div>
    <div class="nbrs" style="margin-top:10px">${s.neighbours.map((n, i) => `
      <div class="nbr"><img src="${esc(n.thumb)}" alt="" loading="lazy" style="outline:3px solid ${NB_COLORS[i]};outline-offset:-3px">
      <div class="c"><b>${esc(n.sample_id)}</b> · ${esc(n.type)}<br>lab G${pct(n.fractions[0])} S${pct(n.fractions[1])} F${pct(n.fractions[2])}</div></div>`).join("")}</div>`;
  document.querySelectorAll(".hm-ctl button").forEach((b) => b.addEventListener("click", () => {
    heat = b.dataset.h;
    document.querySelectorAll(".hm-ctl button").forEach((x) => x.classList.toggle("on", x === b));
    $("#photo").src = heat === "none" ? s.img : s.heat[heat];
    $("#hm-scale").style.display = heat === "none" ? "none" : "block";
  }));
}

function plotSVG(s) {
  const W = 720, H = 300, m = { l: 46, r: 14, t: 22, b: 40 };
  const x0 = Math.log10(100), x1 = Math.log10(0.03);
  const X = (mm) => m.l + (Math.log10(mm) - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = (p) => m.t + (1 - p / 100) * (H - m.t - m.b);
  const sizes = DATA.sieve_mm;
  const line = (ys) => sizes.map((mm, i) => `${i ? "L" : "M"}${X(mm).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
  const zones = [[100, 5, "gravel", "#f2e6d0"], [5, 0.08, "sand", "#eef1e0"], [0.08, 0.03, "fines", "#e3ebf2"]];
  let g = zones.map(([a, b, lab, c]) => `<rect x="${X(a)}" y="${m.t}" width="${X(b) - X(a)}" height="${H - m.t - m.b}" fill="${c}"/>
    <text x="${(X(a) + X(b)) / 2}" y="${m.t - 7}" text-anchor="middle" font-size="12" fill="#6b665d">${lab}</text>`).join("");
  for (const p of [0, 25, 50, 75, 100]) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(p)}" y2="${Y(p)}" stroke="#d9d3c7" stroke-width="0.6"/>
    <text x="${m.l - 6}" y="${Y(p) + 4}" text-anchor="end" font-size="11" fill="#6b665d">${p}</text>`;
  for (const t of [50, 20, 5, 2, 0.5, 0.16, 0.08]) g += `<text x="${X(t)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="#6b665d">${t}</text>`;
  g += `<text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#6b665d">particle size (mm)</text>
    <text x="12" y="${(H - m.b + m.t) / 2}" text-anchor="middle" font-size="12" fill="#6b665d" transform="rotate(-90 12 ${(H - m.b + m.t) / 2})">% passing</text>`;
  s.neighbours.forEach((n, i) => { g += `<path d="${line(n.sieves)}" fill="none" stroke="${NB_COLORS[i]}" stroke-width="1.6" opacity="0.85"/>`; });
  g += `<path d="${line(s.lab_curve)}" fill="none" stroke="#1f1d1a" stroke-width="2.4" stroke-dasharray="6 4"/>`;
  const pts = [[5, s.est.sand + s.est.fines, s.half.gravel], [0.08, s.est.fines, s.half.fines]];
  for (const [mm, p, h] of pts) {
    const hh = Number.isFinite(h) ? h : 0, cx = X(mm);
    g += `<line x1="${cx}" x2="${cx}" y1="${Y(Math.min(100, p + hh))}" y2="${Y(Math.max(0, p - hh))}" stroke="#1f1d1a" stroke-width="2"/>
      <line x1="${cx - 6}" x2="${cx + 6}" y1="${Y(Math.min(100, p + hh))}" y2="${Y(Math.min(100, p + hh))}" stroke="#1f1d1a" stroke-width="2"/>
      <line x1="${cx - 6}" x2="${cx + 6}" y1="${Y(Math.max(0, p - hh))}" y2="${Y(Math.max(0, p - hh))}" stroke="#1f1d1a" stroke-width="2"/>
      <circle cx="${cx}" cy="${Y(p)}" r="5.5" fill="#1f1d1a" stroke="#fff" stroke-width="1.5"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Sieve curves of similar samples with this sample's estimate">${g}</svg>`;
}

function renderCV() {
  const rows = DATA.summary.cv || [];
  const best = rows.filter((r) => !/baseline|nearest|patchmean/.test(r.features)).reduce((a, r) => (!a || r.mae_mean < a.mae_mean ? r : a), null);
  const nice = (f) => ({
    "predict the mean (baseline)": "Always guess the average (baseline)", handcrafted: "Hand-made colour / texture features",
    effnet_b0: "EfficientNet-B0 (ImageNet)", dinov3_convnext_t: "DINOv3 ConvNeXt-T", dinov3_vits16: "DINOv3 ViT-S/16",
    dinov3_both: "DINOv3 ConvNeXt-T + ViT-S (used)", dinov3_vits16_patchmean: "ViT-S patch head (the heatmap)",
  }[f] || f.replace("5 nearest samples", "Average of the 5 most similar samples").replace(/\((dinov3_both|[a-z0-9_]+)\)/, ""));
  $("#cv").innerHTML = `<table class="cv"><tr><th>Features</th><th>Gravel</th><th>Sand</th><th>Fines</th><th>Mean</th></tr>
    ${rows.map((r) => `<tr class="${r === best ? "best" : ""}"><td>${esc(nice(r.features))}</td><td>${pct(r.mae_gravel, 1)}</td><td>${pct(r.mae_sand, 1)}</td><td>${pct(r.mae_fines, 1)}</td><td>${pct(r.mae_mean, 1)}</td></tr>`).join("")}</table>`;
}
