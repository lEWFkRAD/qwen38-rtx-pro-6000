"use strict";

// Strata Lens capture visualization. Consumes /api/viz for the run selected
// in the evidence panel and renders the recorded router trace (48 layers x
// tokens x top-10 experts) plus final-residual activation aggregates.
// Tokens are shown by position only, matching the observer's whitelist:
// no token ids, prompt text, paths or manifest contents reach the browser
// beyond the parsed numbers the server chooses to serve.
(() => {
  const $ = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const expertColor = (id, alpha = 1) => `hsla(${(id * 137.508) % 360}, 72%, 58%, ${alpha})`;
  const GRADIENTS = {
    top1: [[9, 14, 26], [16, 78, 99], [20, 158, 146], [164, 230, 120], [250, 224, 120]],
    entropy: [[9, 14, 26], [68, 32, 110], [150, 52, 110], [226, 110, 60], [250, 210, 110]],
    residual: [[9, 14, 26], [30, 64, 120], [24, 140, 170], [120, 220, 170], [250, 240, 170]],
  };
  const gradient = (stops, t) => {
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  };

  const section = $("section", "viz-section");
  section.id = "viz-section";
  section.hidden = true;
  section.setAttribute("aria-labelledby", "viz-heading");
  const evidence = document.querySelector(".evidence-section");
  evidence.after(section);

  const view = {
    data: null, runId: null, token: 0, layer: 0, metric: "top1",
    playing: false, playMode: "layer", timer: null, speed: 1, loading: false,
  };

  // ---- static skeleton ----------------------------------------------------
  const headingRow = $("div", "section-heading");
  headingRow.append($("h2", "", "Expert routing & activation"), $("span", "section-note", "", ));
  headingRow.lastChild.id = "viz-run-tag";
  headingRow.lastChild.textContent = "RECORDED CAPTURE · SERVER-PARSED";
  const caveat = $("p", "viz-caveat",
    "Router weights are the recorded prefill trace: for each token position and layer, the 10 experts the router selected and their normalized weights. " +
    "The residual panel is the FINAL residual only (one row per token, after the last layer). There is no per-layer residual in this capture, and none is implied.");
  const chips = $("div", "viz-chips");
  chips.id = "viz-chips";

  const controls = $("div", "viz-controls");
  const playButton = $("button", "button viz-play", "▶ Replay capture");
  playButton.type = "button";
  const modeGroup = $("div", "viz-segmented");
  const modeLayer = $("button", "", "Sweep layers"); modeLayer.type = "button";
  const modeToken = $("button", "", "Sweep tokens"); modeToken.type = "button";
  modeGroup.append(modeLayer, modeToken);
  const speedSelect = document.createElement("select");
  speedSelect.className = "viz-select";
  speedSelect.setAttribute("aria-label", "Replay speed");
  for (const [label, value] of [["0.5×", 0.5], ["1×", 1], ["2×", 2], ["4×", 4]]) {
    const opt = document.createElement("option");
    opt.textContent = label; opt.value = value; if (value === 1) opt.selected = true;
    speedSelect.append(opt);
  }
  const tokenCtl = $("label", "viz-slider");
  tokenCtl.append($("span", "", "Token"), slider("viz-token-slider"), $("output", "viz-out"));
  tokenCtl.querySelector("span").id = "viz-token-label";
  const layerCtl = $("label", "viz-slider");
  layerCtl.append($("span", "", "Layer"), slider("viz-layer-slider"), $("output", "viz-out"));
  layerCtl.querySelector("span").id = "viz-layer-label";
  const metricGroup = $("div", "viz-segmented");
  const metricTop1 = $("button", "", "Top-1 weight"); metricTop1.type = "button";
  const metricEntropy = $("button", "", "Entropy"); metricEntropy.type = "button";
  metricGroup.append(metricTop1, metricEntropy);
  const metricLabel = $("span", "viz-control-label", "Map shows");
  controls.append(playButton, modeGroup, speedSelect, tokenCtl, layerCtl, metricLabel, metricGroup);

  function slider(id) {
    const input = document.createElement("input");
    input.type = "range"; input.id = id; input.min = "0"; input.value = "0";
    return input;
  }

  const grid = $("div", "viz-grid");
  const mapCard = card("Layer × token routing map", "Click any cell to inspect it. Colour is the selected metric for that token at that layer.");
  mapCard.body.append(canvasEl("viz-map"));
  mapCard.body.append($("div", "viz-legend", "")); mapCard.body.lastChild.id = "viz-map-legend";
  const cellCard = card("Selected token · layer", "The router's 10 picks for this token position at this layer, in rank order.");
  cellCard.body.id = "viz-cell";
  const stripsCard = card("All 48 layers for the selected token", "One strip per layer, segments sized by routing weight and coloured by expert. Click a strip to jump to that layer.");
  stripsCard.body.append(canvasEl("viz-strips"));
  const loadCard = card("Expert load", "How concentrated the routing is per layer, and which experts the whole capture leaned on.");
  loadCard.body.append(canvasEl("viz-load"));
  loadCard.body.append($("div", "", "")); loadCard.body.lastChild.id = "viz-top-experts";
  loadCard.body.append($("p", "viz-sub", "All 512 experts — brightness is total selections across the capture (log scale). Hover a cell for its expert."));
  loadCard.body.append(canvasEl("viz-experts512"));
  const residualCard = card("Final residual activation", "One row per token from prefill_final_r.bf16. The map shows RMS activation across 64 row buckets; the trace is each token's full-row L2 norm.");
  residualCard.body.append(canvasEl("viz-residual"));
  residualCard.body.append(canvasEl("viz-norms"));
  residualCard.body.append($("div", "", "")); residualCard.body.lastChild.id = "viz-token-stats";
  grid.append(mapCard.node, cellCard.node, stripsCard.node, loadCard.node, residualCard.node);
  section.append(headingRow, caveat, chips, controls, grid);
  const tooltip = $("div", "viz-tip"); tooltip.hidden = true;
  document.body.append(tooltip);

  function card(title, sub) {
    const node = $("article", "viz-card");
    const head = $("div", "viz-card-head");
    head.append($("h3", "", title), $("p", "viz-sub", sub));
    const body = $("div", "viz-card-body");
    node.append(head, body);
    return { node, body };
  }
  function canvasEl(cls) {
    const c = document.createElement("canvas");
    c.className = cls;
    return c;
  }
  const canvas = (cls) => section.querySelector(`.${cls}`);

  // ---- data ----------------------------------------------------------------
  async function load(runId) {
    if (view.loading || (view.runId === runId && view.data)) return;
    view.loading = true;
    view.data = null;
    view.runId = runId;
    try {
      const response = await fetch(`/api/viz?id=${encodeURIComponent(runId)}`, { cache: "no-store", credentials: "same-origin" });
      const data = await response.json();
      if (view.runId !== runId) return;
      if (!data || data.available !== true) {
        section.hidden = false;
        chips.replaceChildren($("span", "viz-chip is-false", data?.reason || "Visualization unavailable for this run."));
        grid.style.display = "none";
        controls.style.display = "none";
        return;
      }
      view.data = data;
      view.token = 0; view.layer = 0;
      grid.style.display = "";
      controls.style.display = "";
      setupControls();
      renderChips();
      renderAll();
      section.hidden = false;
    } catch (_) {
      section.hidden = false;
      chips.replaceChildren($("span", "viz-chip is-false", "Visualization feed unavailable. The capture receipts above are unaffected."));
      grid.style.display = "none";
      controls.style.display = "none";
    } finally {
      view.loading = false;
    }
  }

  window.addEventListener("lens:run-selected", (event) => {
    const { id, status, testFixture, mode } = event.detail || {};
    if (!id) return;
    if (mode === "demo" || testFixture || status !== "recorded_success") {
      stopPlay();
      section.hidden = true;
      view.data = null; view.runId = null;
      return;
    }
    load(id);
  });

  function setupControls() {
    const d = view.data;
    const tokenSlider = document.getElementById("viz-token-slider");
    const layerSlider = document.getElementById("viz-layer-slider");
    tokenSlider.max = String(d.real_tokens - 1);
    layerSlider.max = String(d.layer_count - 1);
    tokenSlider.value = "0"; layerSlider.value = "0";
  }

  function renderChips() {
    const d = view.data;
    const v = d.validation || {};
    const range = Array.isArray(v.routes_weight_sum_range) ? v.routes_weight_sum_range.join(" – ") : "—";
    chips.replaceChildren(
      $("span", "viz-chip", `✓ ${d.real_tokens} tokens × ${d.layer_count} layers × top-${d.top_k} of ${d.experts} experts`),
      $("span", "viz-chip", `✓ Route weight sums ${range}`),
      $("span", "viz-chip", v.residual_all_finite ? "✓ Residual finite · all rows nonzero" : "Residual checks not recorded"),
      $("span", "viz-chip", "Structural parse · not a causal interpretation"),
    );
  }

  // ---- derived values --------------------------------------------------------
  const cellOf = (layer, token) => {
    const ids = view.data.layers[layer].ids[token];
    const weights = view.data.layers[layer].weights[token];
    const total = weights.reduce((a, b) => a + b, 0);
    const entropy = -weights.reduce((a, w) => (w > 0 ? a + (w / total) * Math.log(w / total) : a), 0);
    return { ids, weights, total, entropy, top1: Math.max(...weights) };
  };
  const metricValue = (layer, token) => {
    const cell = cellOf(layer, token);
    if (view.metric === "entropy") return cell.entropy / Math.log(view.data.top_k);
    return cell.top1;
  };

  // ---- rendering -------------------------------------------------------------
  function fitCanvas(c, height) {
    const width = c.clientWidth || c.parentElement.clientWidth || 600;
    const ratio = window.devicePixelRatio || 1;
    c.width = Math.round(width * ratio);
    c.height = Math.round(height * ratio);
    c.style.height = `${height}px`;
    const ctx = c.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width };
  }

  function renderMap() {
    const d = view.data;
    const c = canvas("viz-map");
    const rows = d.layer_count, cols = d.real_tokens;
    const cellH = 9;
    const { ctx, width } = fitCanvas(c, rows * cellH + 26);
    const cellW = (width - 34) / cols;
    ctx.clearRect(0, 0, width, rows * cellH + 26);
    for (let layer = 0; layer < rows; layer++) {
      for (let token = 0; token < cols; token++) {
        ctx.fillStyle = gradient(GRADIENTS[view.metric], metricValue(layer, token));
        ctx.fillRect(34 + token * cellW, layer * cellH, Math.max(1, cellW - 0.4), cellH - 1);
      }
    }
    ctx.fillStyle = "#92a5b8";
    ctx.font = "10px ui-monospace, monospace";
    for (let layer = 0; layer < rows; layer += 8) ctx.fillText(`L${layer}`, 2, layer * cellH + 8);
    ctx.fillText(`L${rows - 1}`, 2, (rows - 1) * cellH + 8);
    ctx.fillText("token →", 34, rows * cellH + 18);
    ctx.strokeStyle = "#e4edf5";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(34 + view.token * cellW - 1, view.layer * cellH - 1, cellW + 2, cellH + 1);
    const legend = document.getElementById("viz-map-legend");
    const label = view.metric === "entropy" ? "routing entropy (spread ↔ focused)" : "top-1 expert weight (diffuse ↔ dominant)";
    legend.textContent = `${view.metric === "entropy" ? "Entropy" : "Top-1 weight"}: low → high · ${label}`;
  }

  function renderCell() {
    const d = view.data;
    const cell = cellOf(view.layer, view.token);
    const host = document.getElementById("viz-cell");
    const head = $("div", "viz-cell-head");
    head.append($("strong", "", `Token ${view.token} · Layer ${view.layer}`),
      $("span", "viz-cell-meta", `weight sum ${cell.total.toFixed(3)} · entropy ${cell.entropy.toFixed(3)} nats · top-1 ${cell.top1.toFixed(3)}`));
    const list = $("div", "viz-bars");
    cell.ids.forEach((expert, rank) => {
      const weight = cell.weights[rank];
      const row = $("div", "viz-bar-row");
      row.append($("span", "viz-rank", `#${rank + 1}`));
      const swatch = $("span", "viz-swatch"); swatch.style.background = expertColor(expert);
      row.append(swatch, $("span", "viz-expert", `E${expert}`));
      const track = $("span", "viz-track");
      const fill = $("span", "viz-fill");
      fill.style.width = `${Math.max(1.5, weight * 100)}%`;
      fill.style.background = expertColor(expert);
      track.append(fill);
      row.append(track, $("span", "viz-weight", `${(weight * 100).toFixed(1)}%`));
      list.append(row);
    });
    const layerStats = d.layer_stats[view.layer];
    const foot = $("p", "viz-sub",
      `Layer ${view.layer} across all ${d.real_tokens} tokens: ${layerStats.distinct_experts} distinct experts used, ` +
      `mean top-1 ${layerStats.top1_mean}, mean entropy ${layerStats.entropy_mean} nats.`);
    host.replaceChildren(head, list, foot);
    document.getElementById("viz-token-label").textContent = `Token ${view.token} / ${d.real_tokens - 1}`;
    document.getElementById("viz-layer-label").textContent = `Layer ${view.layer} / ${d.layer_count - 1}`;
    document.getElementById("viz-token-slider").value = String(view.token);
    document.getElementById("viz-layer-slider").value = String(view.layer);
  }

  function renderStrips() {
    const d = view.data;
    const c = canvas("viz-strips");
    const rowH = 13;
    const { ctx, width } = fitCanvas(c, d.layer_count * rowH + 4);
    ctx.clearRect(0, 0, width, d.layer_count * rowH + 4);
    for (let layer = 0; layer < d.layer_count; layer++) {
      const { ids, weights } = { ids: d.layers[layer].ids[view.token], weights: d.layers[layer].weights[view.token] };
      let x = 30;
      const y = layer * rowH + 2;
      ctx.fillStyle = layer === view.layer ? "#e4edf5" : "#92a5b8";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(`${layer}`, 8, y + 9);
      weights.forEach((w, rank) => {
        const wpx = w * (width - 36);
        ctx.fillStyle = expertColor(ids[rank], 0.9);
        ctx.fillRect(x, y, Math.max(1, wpx - 1), rowH - 4);
        x += wpx;
      });
      if (layer === view.layer) {
        ctx.strokeStyle = "#e4edf5"; ctx.lineWidth = 1;
        ctx.strokeRect(29, y - 1, width - 34, rowH - 2);
      }
    }
  }

  function renderLoad() {
    const d = view.data;
    const c = canvas("viz-load");
    const { ctx, width } = fitCanvas(c, 150);
    const maxDistinct = Math.max(...d.layer_stats.map((s) => s.distinct_experts));
    const maxEntropy = Math.log(d.top_k);
    ctx.clearRect(0, 0, width, 150);
    const plotW = width - 40, base = 128, top = 12;
    ctx.strokeStyle = "#263442"; ctx.beginPath(); ctx.moveTo(34, base); ctx.lineTo(34 + plotW, base); ctx.stroke();
    d.layer_stats.forEach((s, layer) => {
      const x = 34 + (layer / (d.layer_count - 1)) * plotW;
      const h = (s.distinct_experts / maxDistinct) * (base - top);
      ctx.fillStyle = layer === view.layer ? "#c4ee89" : "rgba(110,227,219,0.55)";
      ctx.fillRect(x - 3, base - h, 6, h);
    });
    ctx.strokeStyle = "#e8c27c"; ctx.lineWidth = 1.6; ctx.beginPath();
    d.layer_stats.forEach((s, layer) => {
      const x = 34 + (layer / (d.layer_count - 1)) * plotW;
      const y = base - (s.entropy_mean / maxEntropy) * (base - top);
      if (layer === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#92a5b8"; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("bars: distinct experts per layer · line: mean entropy (amber)", 34, 146);
    const totals = d.global_usage.map((count, id) => ({ id, count })).sort((a, b) => b.count - a.count).slice(0, 10);
    const max = totals[0]?.count || 1;
    const host = document.getElementById("viz-top-experts");
    const title = $("p", "viz-sub", "Most-selected experts across all layers and tokens:");
    const list = $("div", "viz-bars");
    for (const { id, count } of totals) {
      const row = $("div", "viz-bar-row");
      const swatch = $("span", "viz-swatch"); swatch.style.background = expertColor(id);
      const track = $("span", "viz-track");
      const fill = $("span", "viz-fill"); fill.style.width = `${(count / max) * 100}%`; fill.style.background = expertColor(id);
      track.append(fill);
      row.append(swatch, $("span", "viz-expert", `E${id}`), track, $("span", "viz-weight", `${count}×`));
      list.append(row);
    }
    host.replaceChildren(title, list);
    renderExperts512();
  }

  function renderExperts512() {
    const d = view.data;
    const c = canvas("viz-experts512");
    const cols = 32, cell = Math.max(8, Math.floor(((c.clientWidth || 600) - 4) / cols));
    const rows = Math.ceil(d.experts / cols);
    const { ctx } = fitCanvas(c, rows * cell + 2);
    const maxLog = Math.log1p(Math.max(...d.global_usage));
    for (let id = 0; id < d.experts; id++) {
      const x = (id % cols) * cell, y = Math.floor(id / cols) * cell;
      const t = Math.log1p(d.global_usage[id]) / (maxLog || 1);
      ctx.fillStyle = d.global_usage[id] ? gradient(GRADIENTS.top1, 0.15 + 0.85 * t) : "#131e28";
      ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
    }
    c.onmousemove = (event) => {
      const rect = c.getBoundingClientRect();
      const gx = Math.floor((event.clientX - rect.left) / cell);
      const gy = Math.floor((event.clientY - rect.top) / cell);
      const id = gy * cols + gx;
      if (id >= 0 && id < d.experts) showTip(event, `Expert ${id} · selected ${d.global_usage[id]}× in this capture`);
    };
    c.onmouseleave = hideTip;
  }

  function renderResidual() {
    const d = view.data;
    const tokens = d.residual.per_token;
    const cols = tokens.length, rows = d.residual.buckets;
    const c = canvas("viz-residual");
    const cellH = 4;
    const { ctx, width } = fitCanvas(c, rows * cellH + 20);
    const cellW = (width - 34) / cols;
    const maxRms = Math.max(...tokens.flatMap((t) => t.bucket_rms));
    ctx.clearRect(0, 0, width, rows * cellH + 20);
    tokens.forEach((t, token) => {
      t.bucket_rms.forEach((rms, bucket) => {
        ctx.fillStyle = gradient(GRADIENTS.residual, rms / (maxRms || 1));
        ctx.fillRect(34 + token * cellW, bucket * cellH, Math.max(1, cellW - 0.4), cellH - 0.6);
      });
    });
    ctx.fillStyle = "#92a5b8"; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("bucket →", 34, rows * cellH + 15);
    ctx.fillRect(34 + view.token * cellW - 1, 0, 1.6, rows * cellH);
    const n = canvas("viz-norms");
    const fit = fitCanvas(n, 90);
    const norms = tokens.map((t) => t.norm);
    const maxNorm = Math.max(...norms);
    fit.ctx.clearRect(0, 0, fit.width, 90);
    fit.ctx.strokeStyle = "#263442"; fit.ctx.beginPath(); fit.ctx.moveTo(34, 76); fit.ctx.lineTo(fit.width - 4, 76); fit.ctx.stroke();
    fit.ctx.strokeStyle = "#6ee3db"; fit.ctx.lineWidth = 1.6; fit.ctx.beginPath();
    norms.forEach((norm, token) => {
      const x = 34 + (token / (cols - 1)) * (fit.width - 40);
      const y = 76 - (norm / (maxNorm || 1)) * 62;
      if (token === 0) fit.ctx.moveTo(x, y); else fit.ctx.lineTo(x, y);
    });
    fit.ctx.stroke();
    const sx = 34 + (view.token / (cols - 1)) * (fit.width - 40);
    fit.ctx.fillStyle = "#e8c27c";
    fit.ctx.beginPath(); fit.ctx.arc(sx, 76 - (norms[view.token] / (maxNorm || 1)) * 62, 3.5, 0, Math.PI * 2); fit.ctx.fill();
    fit.ctx.fillStyle = "#92a5b8"; fit.ctx.font = "10px ui-monospace, monospace";
    fit.ctx.fillText(`token L2 norm · max ${maxNorm.toFixed(1)}`, 34, 10);
    const t = tokens[view.token];
    const host = document.getElementById("viz-token-stats");
    const streams = t.stream_norms.map((s, i) => `block ${i + 1}: ${s}`).join(" · ");
    host.replaceChildren(
      $("p", "viz-sub", `Token ${view.token} final residual: L2 norm ${t.norm} · mean ${t.mean} · max |value| ${t.max_abs}`),
      $("p", "viz-sub", `Row-quarter (HC stream block) norms — ${streams}`),
    );
  }

  function renderAll() {
    if (!view.data) return;
    renderMap(); renderCell(); renderStrips(); renderLoad(); renderResidual();
  }

  // ---- interaction -----------------------------------------------------------
  function select(token, layer, rerender = true) {
    const d = view.data;
    view.token = Math.max(0, Math.min(d.real_tokens - 1, token));
    view.layer = Math.max(0, Math.min(d.layer_count - 1, layer));
    if (rerender) renderAll();
  }
  function showTip(event, text) {
    tooltip.textContent = text;
    tooltip.hidden = false;
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 12}px`;
  }
  function hideTip() { tooltip.hidden = true; }

  canvas("viz-map").addEventListener("click", (event) => {
    const d = view.data; if (!d) return;
    const rect = event.target.getBoundingClientRect();
    const cellW = (rect.width - 34) / d.real_tokens;
    const token = Math.floor((event.clientX - rect.left - 34) / cellW);
    const layer = Math.floor((event.clientY - rect.top) / 9);
    if (token >= 0 && token < d.real_tokens && layer >= 0 && layer < d.layer_count) select(token, layer);
  });
  canvas("viz-map").addEventListener("mousemove", (event) => {
    const d = view.data; if (!d) return;
    const rect = event.target.getBoundingClientRect();
    const cellW = (rect.width - 34) / d.real_tokens;
    const token = Math.floor((event.clientX - rect.left - 34) / cellW);
    const layer = Math.floor((event.clientY - rect.top) / 9);
    if (token >= 0 && token < d.real_tokens && layer >= 0 && layer < d.layer_count) {
      const cell = cellOf(layer, token);
      showTip(event, `Token ${token} · Layer ${layer} · top-1 E${cell.ids[cell.weights.indexOf(cell.top1)]} ${(cell.top1 * 100).toFixed(1)}% · entropy ${cell.entropy.toFixed(2)}`);
    } else hideTip();
  });
  canvas("viz-map").addEventListener("mouseleave", hideTip);
  canvas("viz-strips").addEventListener("click", (event) => {
    const d = view.data; if (!d) return;
    const rect = event.target.getBoundingClientRect();
    const layer = Math.floor((event.clientY - rect.top) / 13);
    if (layer >= 0 && layer < d.layer_count) select(view.token, layer);
  });
  document.getElementById("viz-token-slider").addEventListener("input", (e) => select(Number(e.target.value), view.layer));
  document.getElementById("viz-layer-slider").addEventListener("input", (e) => select(view.token, Number(e.target.value)));
  metricTop1.addEventListener("click", () => { view.metric = "top1"; syncSegments(); renderMap(); });
  metricEntropy.addEventListener("click", () => { view.metric = "entropy"; syncSegments(); renderMap(); });
  modeLayer.addEventListener("click", () => { view.playMode = "layer"; syncSegments(); });
  modeToken.addEventListener("click", () => { view.playMode = "token"; syncSegments(); });
  speedSelect.addEventListener("change", () => { view.speed = Number(speedSelect.value); if (view.playing) startPlay(); });
  function syncSegments() {
    for (const [button, active] of [[metricTop1, view.metric === "top1"], [metricEntropy, view.metric === "entropy"],
                                    [modeLayer, view.playMode === "layer"], [modeToken, view.playMode === "token"]]) {
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  }
  syncSegments();

  function startPlay() {
    stopPlay(false);
    view.playing = true;
    playButton.textContent = "⏸ Pause replay";
    view.timer = setInterval(() => {
      if (!view.data) return stopPlay();
      if (view.playMode === "layer") {
        select(view.token, (view.layer + 1) % view.data.layer_count);
        if (view.layer === 0 && view.token < view.data.real_tokens - 1) select(view.token + 1, 0);
      } else {
        select((view.token + 1) % view.data.real_tokens, view.layer);
      }
    }, Math.round(160 / view.speed));
  }
  function stopPlay(resetLabel = true) {
    view.playing = false;
    if (view.timer) clearInterval(view.timer);
    view.timer = null;
    if (resetLabel) playButton.textContent = "▶ Replay capture";
  }
  playButton.addEventListener("click", () => (view.playing ? stopPlay() : startPlay()));
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (view.data && !section.hidden) renderAll(); }, 180);
  });
})();
