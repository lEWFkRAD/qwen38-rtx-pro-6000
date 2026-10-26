"use strict";

// This view consumes summaries only. It never invokes a model or changes a run.
(() => {
  const byId = (id) => document.getElementById(id);
  const state = { data: null, selectedId: null, paused: false, fetching: false,
    lastSuccess: null, failed: false, runFingerprint: "", detailFingerprint: "" };
  const statuses = {
    recorded_success: { label: "Recorded success", tone: "success" },
    failed: { label: "Failed", tone: "failed" },
    in_progress: { label: "In progress", tone: "progress" },
    unreadable: { label: "Unreadable", tone: "failed" },
    demo: { label: "Synthetic demo", tone: "demo" },
  };
  const number = (value) => Number.isFinite(value) && value >= 0
    ? value.toLocaleString("en-US") : "—";
  const valueText = (value, fallback = "Not recorded") =>
    typeof value === "string" && value ? value : fallback;
  const statusOf = (run) => state.data?.mode === "demo"
    ? { label: "Synthetic demo", tone: "demo" }
    : run.test_fixture ? { label: "Test fixture", tone: "demo" }
    : statuses[run.status] || { label: "Unknown", tone: "" };
  const element = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = String(content);
    return node;
  };
  const timeText = (value) => {
    const date = typeof value === "string" ? new Date(value) : null;
    return date && !Number.isNaN(date.getTime())
      ? date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
      : "Time not recorded";
  };
  const bytesText = (value) => {
    if (!Number.isFinite(value) || value < 0) return "Size not recorded";
    if (value < 1024) return `${number(value)} B`;
    if (value < 1048576) return `${(value / 1024).toFixed(2)} KiB`;
    return `${(value / 1048576).toFixed(2)} MiB`;
  };

  function updateConnection() {
    const demo = state.data?.mode === "demo";
    const age = state.lastSuccess ? Math.max(0, Math.floor((Date.now() - state.lastSuccess) / 1000)) : null;
    const stale = age !== null && age >= 10;
    let label = "Connecting to local observer";
    let tone = "";
    if (state.failed) { label = "Observer disconnected"; tone = "is-error"; }
    else if (state.paused) { label = "Updates paused"; }
    else if (state.data) { label = demo ? "Demo feed · synthetic" : "Connected · read-only"; tone = demo ? "is-demo" : "is-live"; }
    if (stale && !state.paused && !state.failed) { label = "Waiting for fresh data"; tone = ""; }
    if (byId("connection-label").textContent !== label) byId("connection-label").textContent = label;
    byId("connection-dot").className = `connection-dot ${tone}`;
    byId("freshness").textContent = age === null ? "No observer data received"
      : `Last received ${age === 0 ? "just now" : `${age}s ago`}${stale ? " · STALE" : ""}`;
    const warning = byId("connection-warning");
    warning.hidden = !state.failed;
    warning.textContent = state.data
      ? "The observer cannot be reached. Displayed receipts are from the last successful response; they are not current."
      : "The local observer cannot be reached. No live data is shown. Check that the server is running, then refresh.";
    byId("pause-button").textContent = state.paused ? "Resume updates" : "Pause updates";
    byId("pause-button").setAttribute("aria-pressed", String(state.paused));
  }

  function renderPlan(plan) {
    if (!plan || typeof plan !== "object") return;
    const metric = (id, value, unit) => {
      const node = byId(id);
      node.replaceChildren(document.createTextNode(value), element("span", "metric-unit", unit));
    };
    metric("plan-tokens", number(plan.real_tokens), "tokens");
    metric("plan-layers", number(plan.layers), "layers");
    metric("plan-experts", number(plan.experts_per_token), "of 512");
    metric("plan-residual", Number.isFinite(plan.residual_bytes) ? (plan.residual_bytes / 1048576).toFixed(2) : "—", "MiB");
    byId("plan-launch").textContent = `${number(plan.launch_tokens)} at launch · includes sentinel`;
    byId("plan-prompt").textContent = valueText(plan.prompt_id, "Reference prompt");
    byId("plan-routes").textContent = `${number(plan.routes_bytes)} bytes`;
  }

  function selectRun(id) {
    state.selectedId = id;
    for (const button of byId("run-list").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.runId === id));
    }
    renderDetail();
  }

  function renderRuns(runs) {
    const fingerprint = JSON.stringify([state.data.mode, runs.map(({ id, status, prompt_id, real_tokens, started_at, updated_at, test_fixture }) =>
      ({ id, status, prompt_id, real_tokens, started_at, updated_at, test_fixture }))]);
    if (fingerprint !== state.runFingerprint) {
      const focusedId = document.activeElement?.dataset?.runId;
      const fragment = document.createDocumentFragment();
      for (const run of runs) {
        const status = statusOf(run);
        const item = element("li");
        const button = element("button", "run-button");
        button.type = "button";
        button.dataset.runId = run.id;
        button.setAttribute("aria-pressed", String(run.id === state.selectedId));
        button.setAttribute("aria-label", `${valueText(run.prompt_id, run.id)} — ${status.label} — ${run.id}`);
        const top = element("span", "run-button-top");
        top.append(element("span", "run-name", valueText(run.prompt_id, run.id)));
        const dot = element("span", `status-dot tone-${status.tone}`);
        dot.setAttribute("aria-hidden", "true");
        top.append(dot);
        const sub = element("span", "run-button-sub");
        sub.append(element("span", `tone-${status.tone}`, status.label), element("span", "", timeText(run.updated_at || run.started_at)));
        button.append(top, sub);
        button.addEventListener("click", () => selectRun(run.id));
        item.append(button);
        fragment.append(item);
      }
      byId("run-list").replaceChildren(fragment);
      if (focusedId) {
        const replacement = Array.from(byId("run-list").querySelectorAll("button"))
          .find((button) => button.dataset.runId === focusedId);
        replacement?.focus({ preventScroll: true });
      }
      state.runFingerprint = fingerprint;
    }
    byId("run-count").textContent = number(runs.length);
    byId("run-empty").hidden = runs.length > 0;
    byId("run-feed-label").textContent = state.data.mode === "demo" ? "SYNTHETIC" : "LOCAL RECEIPTS";
    renderDetail();
  }

  function renderDetail() {
    const run = state.data?.runs.find((item) => item.id === state.selectedId);
    const detail = byId("run-detail");
    byId("detail-placeholder").hidden = Boolean(run);
    detail.hidden = !run;
    if (!run) { state.detailFingerprint = ""; detail.replaceChildren(); return; }
    window.dispatchEvent(new CustomEvent("lens:run-selected", { detail: { id: run.id, status: run.status, testFixture: run.test_fixture === true, mode: state.data?.mode } }));
    const fingerprint = JSON.stringify([state.data.mode, run]);
    if (fingerprint === state.detailFingerprint) return;
    state.detailFingerprint = fingerprint;
    const status = statusOf(run);
    const synthetic = state.data.mode === "demo" || run.status === "demo" || run.test_fixture === true;
    const header = element("div", "detail-header");
    const kicker = element("div", "detail-kicker");
    kicker.append(element("p", "eyebrow", run.test_fixture ? "TEST FIXTURE / NOT A GPU CAPTURE" : synthetic ? "SYNTHETIC EXAMPLE / NOT EVIDENCE" : "RECORDED MANIFEST / NOT REVALIDATED"),
      element("span", `status-pill tone-${status.tone}`, status.label));
    header.append(kicker, element("h3", "detail-title", run.id));
    const summary = element("div", "detail-summary");
    const fields = [["Prompt", valueText(run.prompt_id)], ["Real tokens", number(run.real_tokens)],
      ["Duration", Number.isFinite(run.duration_s) ? `${run.duration_s.toFixed(2)}s` : "Not recorded"],
      ["Manifest updated", timeText(run.updated_at)]];
    for (const [label, value] of fields) {
      const item = element("span", "", `${label} `);
      item.append(element("strong", "", value));
      summary.append(item);
    }
    header.append(summary);
    if (run.message) header.append(element("p", "detail-message", String(run.message)));
    const body = element("div", "detail-body");
    const checkHeading = element("div", "detail-subheading");
    checkHeading.append(element("h3", "", "Recorded checks"), element("span", "", "MANIFEST VALUES"));
    body.append(checkHeading);
    const checkLabels = {
      success_recorded: "Manifest success", input_attestation_recorded: "Input attestation recorded",
      cleanup_recorded: "Cleanup recorded", durability_recorded: "Durability recorded",
      artifact_sizes_match: "Artifact sizes match",
    };
    const checks = Object.entries(run.checks || {}).filter(([key, value]) =>
      Object.hasOwn(checkLabels, key) && (typeof value === "boolean" || value === null));
    if (checks.length) {
      const grid = element("div", "checks-grid");
      for (const [key, value] of checks) {
        const chip = element("span", `check-chip${value === false ? " is-false" : value === null ? " is-unknown" : ""}`);
        const humanKey = checkLabels[key];
        chip.append(element("span", "check-indicator", value === null ? "?" : value ? "✓" : "×"), document.createTextNode(humanKey));
        chip.setAttribute("aria-label", `${humanKey}: ${value === null ? "unknown" : value ? "true" : "false"}`);
        grid.append(chip);
      }
      body.append(grid);
    } else body.append(element("p", "no-data", "No supported check values in this receipt."));
    const artifactHeading = element("div", "detail-subheading");
    const artifacts = Array.isArray(run.artifacts) ? run.artifacts : [];
    artifactHeading.append(element("h3", "", "Artifact receipts"), element("span", "", `${artifacts.length} RECORDED`));
    body.append(artifactHeading);
    if (artifacts.length) {
      const list = element("ul", "artifacts");
      for (const artifact of artifacts) {
        const item = element("li", "artifact");
        const top = element("div", "artifact-top");
        top.append(element("span", "artifact-name", valueText(artifact.name, "Unnamed artifact")),
          element("span", "artifact-bytes", `${bytesText(artifact.bytes)} recorded`));
        const hash = element("p", "artifact-hash");
        hash.append(element("span", "artifact-hash-label", "SHA-256"),
          document.createTextNode(valueText(artifact.sha256, "Not recorded")));
        item.append(top, hash);
        const sizeLabel = artifact.size_matches === true ? "size matches" : artifact.size_matches === false ? "SIZE MISMATCH" : "size not verified";
        const observed = element("p", `artifact-observed${artifact.size_matches === false ? " tone-failed" : ""}`,
          `On disk: ${bytesText(artifact.observed_bytes)} · ${sizeLabel}`);
        item.append(observed);
        list.append(item);
      }
      body.append(list);
    } else body.append(element("p", "no-data", "No artifact receipts recorded yet."));
    body.append(element("p", "detail-caveat", synthetic
      ? "Synthetic fixture only. Check flags, sizes, and hashes here do not describe a real GPU capture or provide model evidence."
      : "Success, attestation, cleanup, durability, and hashes are manifest claims. Only artifact sizes are compared with on-disk file metadata. This observer does not rehash binaries, verify model inputs, or independently validate a GPU capture. Use the harness validation command with its reviewed profile and contract. “In progress” is a recorded state, not proof of an active engine."));
    detail.replaceChildren(header, body);
  }

  function render(data) {
    state.data = data;
    if (!data.runs.some((run) => run.id === state.selectedId)) state.selectedId = data.runs[0]?.id || null;
    byId("version").textContent = `${valueText(data.version, "preview")} / PREVIEW`;
    byId("demo-banner").hidden = data.mode !== "demo";
    const warnings = Array.isArray(data.warnings) ? data.warnings.filter((item) => typeof item === "string") : [];
    if (data.truncated) warnings.push("Only the server's bounded selection of runs is displayed; additional records may exist.");
    byId("server-warnings").hidden = warnings.length === 0;
    byId("server-warnings").textContent = warnings.join(" ");
    renderPlan(data.plan);
    renderRuns(data.runs);
    updateConnection();
  }

  async function refresh() {
    if (state.fetching) return;
    state.fetching = true;
    byId("refresh-button").disabled = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch("/api/state", { cache: "no-store", credentials: "same-origin", signal: controller.signal });
      if (!response.ok) throw new Error("Observer response unavailable");
      const data = await response.json();
      if (!data || !Array.isArray(data.runs) || !["live", "demo"].includes(data.mode)) throw new Error("Invalid observer response");
      if (!data.runs.every((run) => run && typeof run.id === "string" && run.id.length > 0)) throw new Error("Invalid run summaries");
      state.lastSuccess = Date.now();
      state.failed = false;
      render(data);
    } catch (_) {
      state.failed = true;
      updateConnection();
    } finally {
      clearTimeout(timeout);
      state.fetching = false;
      byId("refresh-button").disabled = false;
    }
  }

  byId("pause-button").addEventListener("click", () => {
    state.paused = !state.paused;
    updateConnection();
    if (!state.paused) refresh();
  });
  byId("refresh-button").addEventListener("click", refresh);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.paused) refresh();
  });
  setInterval(() => { if (!state.paused && !document.hidden) refresh(); }, 2000);
  setInterval(updateConnection, 1000);
  refresh();
})();
