import {
  loadImage,
  fitContain,
  imageToLuminanceArray,
  makeColormap,
  drawHeatmapToCanvas
} from "./image-utils.js";

const els = {
  appTitle: document.getElementById("appTitle"),
  tileSelect: document.getElementById("tileSelect"),
  classSelect: document.getElementById("classSelect"),
  methodTabs: document.getElementById("methodTabs"),
  status: document.getElementById("status"),
  canvas: document.getElementById("viewer"),
  lensRadius: document.getElementById("lensRadius"),
  confidenceMeaning: document.getElementById("confidenceMeaning"),
  hist: document.getElementById("hist"),
  reloadBtn: document.getElementById("reloadBtn"),
  modeLens: document.getElementById("modeLens"),
  modeFull: document.getElementById("modeFull")
};

const state = {
  manifest: null,
  tileId: null,
  classId: 1,
  methodId: "confidence",
  lens: {
    x: 0,
    y: 0,
    r: 80,
    isInside: false
  },
  // histogram brush selection in [0,1]
  riskRange: [0, 0.3],
  viewMode: "lens" // "lens" | "full"
};

const ctx = els.canvas.getContext("2d");

// Cache of loaded images for current tile
let cache = {
  input: null,
  prediction: null,
  confidence: null,
  overlay: null, // current method image
  overlayKey: null,
  confidenceLum: null,
  // cache-busting query suffix like "?v=..."
  bust: ""
};

function setStatus(msg) {
  els.status.textContent = msg;
}

function resizeCanvasToDisplaySize() {
  const { width, height } = els.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(width * dpr));
  const h = Math.max(1, Math.floor(height * dpr));
  if (els.canvas.width !== w || els.canvas.height !== h) {
    els.canvas.width = w;
    els.canvas.height = h;
  }
}

function getTile() {
  return state.manifest.tiles.find((t) => t.id === state.tileId);
}

function getMethod() {
  return state.manifest.methods.find((m) => m.id === state.methodId);
}

function getClassFileKey(classId) {
  const c = state.manifest?.classes?.find((x) => x.id === classId);
  return c?.fileKey || c?.name || String(classId);
}

function getOverlayUrl(tile, methodId, classId) {
  if (!tile) return null;

  // Confidence is global (all classes combined)
  if (methodId === "confidence") {
    // If manifest provides explicit path, use it; otherwise fall back to default name.
    return tile.xai?.confidence || `assets/tiles/${tile.id}/xai/confidence.png`;
  }

  // Class-wise methods
  // New convention (no subfolders): xai/<method>_<classKey>.png
  const classKey = getClassFileKey(classId);
  const auto = `assets/tiles/${tile.id}/xai/${methodId}_${classKey}.png`;

  // Backward compat: optional explicit path at xai.<method>[classId]
  const byMethod = tile.xai?.[methodId];
  const explicit = byMethod ? byMethod[String(classId)] : null;

  return explicit || auto;
}

async function loadManifest() {
  const res = await fetch("assets/manifest.json");
  if (!res.ok) throw new Error("Failed to load assets/manifest.json");
  const manifest = await res.json();
  state.manifest = manifest;
  cache.bust = `?v=${encodeURIComponent(manifest.assetVersion || Date.now())}`;
  els.appTitle.textContent = manifest.appTitle || "Segmentation Detective";

  // init defaults
  state.tileId = manifest.tiles[0]?.id ?? null;
  state.classId = manifest.classes.find((c) => c.id !== 0)?.id ?? 1;
  state.methodId = manifest.methods[0]?.id ?? "confidence";
  state.viewMode = state.methodId === "confidence" ? "full" : "lens";

  buildControls();
  await loadTileAssets();
}

function buildControls() {
  // tiles
  els.tileSelect.innerHTML = "";
  for (const t of state.manifest.tiles) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.label || t.id;
    els.tileSelect.appendChild(opt);
  }
  els.tileSelect.value = state.tileId;

  // classes
  els.classSelect.innerHTML = "";
  for (const c of state.manifest.classes) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.id}: ${c.name}`;
    els.classSelect.appendChild(opt);
  }
  els.classSelect.value = state.classId;

  // tabs
  els.methodTabs.innerHTML = "";
  for (const m of state.manifest.methods) {
    const div = document.createElement("div");
    div.className = "tab";
    div.dataset.methodId = m.id;
    div.textContent = m.label;
    els.methodTabs.appendChild(div);
  }
  setActiveTab();

  els.tileSelect.addEventListener("change", async (e) => {
    state.tileId = e.target.value;
    await loadTileAssets();
  });

  els.reloadBtn?.addEventListener("click", async () => {
    // Force cache bust to current time, reload current tile
    cache.bust = `?v=${Date.now()}`;
    await loadTileAssets();
  });

  els.classSelect.addEventListener("change", async (e) => {
    state.classId = Number(e.target.value);
    await loadOverlay();
    setActiveTab();
    render();
  });

  els.methodTabs.addEventListener("click", async (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    state.methodId = tab.dataset.methodId;

    // Default UX: confidence is usually easiest to interpret as a full overlay.
    state.viewMode = state.methodId === "confidence" ? "full" : "lens";

    setActiveTab();
    await loadOverlay();
    render();
  });

  els.modeLens?.addEventListener("click", async () => {
    state.viewMode = "lens";
    setActiveTab();
    await loadOverlay();
    render();
  });
  els.modeFull?.addEventListener("click", async () => {
    state.viewMode = "full";
    setActiveTab();
    await loadOverlay();
    render();
  });

  // magic lens interactions
  els.canvas.addEventListener("mousemove", (e) => {
    const rect = els.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.lens.x = (e.clientX - rect.left) * dpr;
    state.lens.y = (e.clientY - rect.top) * dpr;
    state.lens.isInside = true;
    render();
  });
  els.canvas.addEventListener("mouseleave", () => {
    state.lens.isInside = false;
    render();
  });
  els.canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const delta = Math.sign(e.deltaY);
      state.lens.r = Math.max(20, Math.min(260, state.lens.r + delta * 10));
      els.lensRadius.textContent = `${state.lens.r} px`;
      render();
    },
    { passive: false }
  );
}

function setActiveTab() {
  const method = getMethod();
  // disable class dropdown if method doesn't use class
  els.classSelect.disabled = !(method?.usesClass ?? true);

  for (const tab of els.methodTabs.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.methodId === state.methodId);
  }

  els.modeLens?.classList.toggle("active", state.viewMode === "lens");
  els.modeFull?.classList.toggle("active", state.viewMode === "full");
}

async function loadTileAssets() {
  const tile = getTile();
  if (!tile) return;

  setStatus("Loading images…");
  els.confidenceMeaning.textContent = tile.confidenceMeaning || "confidence";

  const warnings = [];

  // Reset cache
  cache = { input: null, prediction: null, confidence: null, overlay: null, overlayKey: null, confidenceLum: null, bust: cache.bust };

  // Required
  cache.input = await loadImage(tile.base.input + cache.bust);

  // Optional (don’t fail the whole app if missing)
  if (tile.base.prediction) {
    try {
      cache.prediction = await loadImage(tile.base.prediction + cache.bust);
    } catch (e) {
      warnings.push("prediction missing");
      cache.prediction = null;
    }
  }

  // Confidence is a global method; if present, build histogram.
  // Support both explicit manifest path and default naming.
  const confUrl = tile.xai?.confidence || `assets/tiles/${tile.id}/xai/confidence.png`;
  try {
    cache.confidence = await loadImage(confUrl + cache.bust);
  } catch (e) {
    warnings.push("confidence missing");
    cache.confidence = null;
  }

  await loadOverlay();

  // Build histogram from confidence image (if present)
  if (cache.confidence) {
    cache.confidenceLum = imageToLuminanceArray(cache.confidence);
    buildHistogram(cache.confidenceLum.values);
  } else {
    // clear histogram if confidence missing
    d3.select(els.hist).selectAll("*").remove();
  }

  setStatus(warnings.length ? `Ready (${warnings.join(", ")})` : "Ready");
  render();
}

async function loadOverlay() {
  const tile = getTile();
  const method = getMethod();
  if (!tile || !method) return;

  const overlayUrl = getOverlayUrl(tile, state.methodId, state.classId);

  const key = `${state.tileId}:${state.methodId}:${state.classId}`;
  if (cache.overlayKey === key) return;

  cache.overlay = null;
  cache.overlayKey = key;

  if (!overlayUrl) {
    // If missing, keep overlay null and show info
    setStatus(`Missing overlay: ${state.methodId}${method.usesClass ? ` (class ${state.classId})` : ""}`);
    return;
  }

  try {
    cache.overlay = await loadImage(overlayUrl + cache.bust);
    setStatus("Ready");
  } catch (e) {
    setStatus(String(e.message || e));
  }
}

function render() {
  resizeCanvasToDisplaySize();

  const tile = getTile();
  if (!tile || !cache.input) {
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    return;
  }

  const dst = fitContain(cache.input.naturalWidth, cache.input.naturalHeight, els.canvas.width, els.canvas.height);

  // base: input
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.drawImage(cache.input, dst.x, dst.y, dst.w, dst.h);

  // overlay: prediction (global, lightly)
  if (cache.prediction) {
    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.drawImage(cache.prediction, dst.x, dst.y, dst.w, dst.h);
    ctx.restore();
  }

  // overlay: XAI via lens (clipped circle)
  const method = getMethod();
  if (cache.overlay && state.viewMode === "full") {
    if (method.kind === "grayscale" || method.kind === "heatmap") {
      const colormap = makeColormap(method.colormap);
      drawHeatmapToCanvas(ctx, cache.overlay, dst, { alpha: 0.75, colormap });
    } else {
      ctx.save();
      ctx.globalAlpha = 0.65;
      ctx.drawImage(cache.overlay, dst.x, dst.y, dst.w, dst.h);
      ctx.restore();
    }
  } else if (cache.overlay && state.lens.isInside) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(state.lens.x, state.lens.y, state.lens.r * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.clip();

    if (method.kind === "grayscale" || method.kind === "heatmap") {
      const colormap = makeColormap(method.colormap);
      drawHeatmapToCanvas(ctx, cache.overlay, dst, { alpha: 0.75, colormap });
    } else {
      ctx.globalAlpha = 0.65;
      ctx.drawImage(cache.overlay, dst.x, dst.y, dst.w, dst.h);
    }

    ctx.restore();

    // lens outline
    ctx.save();
    ctx.strokeStyle = "rgba(231,235,255,0.65)";
    ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
    ctx.beginPath();
    ctx.arc(state.lens.x, state.lens.y, state.lens.r * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Debug (small overlay status)
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(dst.x + 8, dst.y + 8, 230, 44);
  ctx.fillStyle = "rgba(231,235,255,0.92)";
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px system-ui`;
  ctx.fillText(`Method: ${state.methodId}`, dst.x + 16, dst.y + 28);
  ctx.fillText(`Mode: ${state.viewMode} | overlay: ${cache.overlay ? "yes" : "no"}` , dst.x + 16, dst.y + 46);
  ctx.restore();

  // “risky pixels” highlight inside lens based on histogram brush
  // This is intentionally lightweight: sample a grid instead of all pixels.
  if (cache.confidence && cache.confidenceLum && state.lens.isInside) {
    const meaning = tile.confidenceMeaning || "confidence";
    const [lo, hi] = state.riskRange; // interpret as confidence range

    // map lens circle (canvas space) to image pixel space
    const dpr = window.devicePixelRatio || 1;
    const cx = state.lens.x;
    const cy = state.lens.y;
    const r = state.lens.r * dpr;

    // Create a sampling step (in canvas pixels)
    const step = Math.max(4 * dpr, Math.floor(r / 22));

    ctx.save();
    ctx.fillStyle = "rgba(255, 60, 60, 0.55)";

    for (let y = cy - r; y <= cy + r; y += step) {
      for (let x = cx - r; x <= cx + r; x += step) {
        const dx = x - cx;
        const dy = y - cy;
        if (dx * dx + dy * dy > r * r) continue;

        // map (x,y) in canvas to (u,v) in image pixel
        const u = (x - dst.x) / dst.w;
        const v = (y - dst.y) / dst.h;
        if (u < 0 || u > 1 || v < 0 || v > 1) continue;

        const px = Math.floor(u * (cache.confidenceLum.width - 1));
        const py = Math.floor(v * (cache.confidenceLum.height - 1));
        const idx = py * cache.confidenceLum.width + px;
        let conf = cache.confidenceLum.values[idx];
        if (meaning === "uncertainty") conf = 1 - conf;

        if (conf >= lo && conf <= hi) {
          ctx.fillRect(x - step / 2, y - step / 2, step, step);
        }
      }
    }

    ctx.restore();
  }
}

function buildHistogram(values01) {
  // values01: Float32 [0,1]
  const binsCount = 30;
  const bins = new Array(binsCount).fill(0);

  for (let i = 0; i < values01.length; i += 5) {
    // subsample for speed
    const v = values01[i];
    const b = Math.max(0, Math.min(binsCount - 1, Math.floor(v * binsCount)));
    bins[b] += 1;
  }

  const svg = d3.select(els.hist);
  svg.selectAll("*").remove();

  const W = els.hist.clientWidth || 900;
  const H = 200;
  const margin = { top: 10, right: 10, bottom: 28, left: 40 };

  svg.attr("viewBox", `0 0 ${W} ${H}`);

  const x = d3.scaleLinear().domain([0, 1]).range([margin.left, W - margin.right]);
  const y = d3
    .scaleLinear()
    .domain([0, d3.max(bins) || 1])
    .nice()
    .range([H - margin.bottom, margin.top]);

  const barW = (x(1) - x(0)) / binsCount;
  svg
    .append("g")
    .selectAll("rect")
    .data(bins.map((count, i) => ({ i, count })))
    .join("rect")
    .attr("x", (d) => x(d.i / binsCount) + 1)
    .attr("y", (d) => y(d.count))
    .attr("width", barW - 2)
    .attr("height", (d) => y(0) - y(d.count))
    .attr("fill", "rgba(122,162,255,0.55)")
    .attr("stroke", "rgba(231,235,255,0.10)");

  svg
    .append("g")
    .attr("transform", `translate(0,${H - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(10))
    .call((g) => g.selectAll("text").attr("fill", "#e7ebff"))
    .call((g) => g.selectAll("path,line").attr("stroke", "rgba(231,235,255,0.25)"));

  svg
    .append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(4))
    .call((g) => g.selectAll("text").attr("fill", "#e7ebff"))
    .call((g) => g.selectAll("path,line").attr("stroke", "rgba(231,235,255,0.25)"));

  svg
    .append("text")
    .attr("x", W / 2)
    .attr("y", H - 8)
    .attr("text-anchor", "middle")
    .attr("fill", "rgba(231,235,255,0.75)")
    .attr("font-size", 12)
    .text("Confidence (0 → low, 1 → high)");

  // Brush to set riskRange
  const brush = d3
    .brushX()
    .extent([
      [margin.left, margin.top],
      [W - margin.right, H - margin.bottom]
    ])
    .on("brush end", ({ selection }) => {
      if (!selection) return;
      const [x0, x1] = selection;
      const lo = x.invert(x0);
      const hi = x.invert(x1);
      state.riskRange = [Math.max(0, lo), Math.min(1, hi)];
      render();
    });

  const brushG = svg.append("g").attr("class", "brush").call(brush);

  // Default brush selection
  const [lo, hi] = state.riskRange;
  brushG.call(brush.move, [x(lo), x(hi)]);
}

window.addEventListener("resize", () => render());

loadManifest().catch((e) => {
  console.error(e);
  setStatus(String(e.message || e));
});
