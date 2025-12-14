export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    // Safe for same-origin http.server. If you later host elsewhere with CORS, keep anonymous.
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

export function fitContain(srcW, srcH, dstW, dstH) {
  const srcRatio = srcW / srcH;
  const dstRatio = dstW / dstH;

  let w, h;
  if (srcRatio > dstRatio) {
    w = dstW;
    h = dstW / srcRatio;
  } else {
    h = dstH;
    w = dstH * srcRatio;
  }

  const x = (dstW - w) / 2;
  const y = (dstH - h) / 2;

  return { x, y, w, h };
}

export function imageToLuminanceArray(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Return Float32 0..1 luminance
  const out = new Float32Array(canvas.width * canvas.height);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    // Standard luminance weights
    out[p] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  return { width: canvas.width, height: canvas.height, values: out };
}

export function makeColormap(name) {
  // d3-scale-chromatic exports interpolators globally if loaded via script tag.
  // We'll accept either a direct interpolator function or a known name.
  const cmap = {
    magma: d3.interpolateMagma,
    inferno: d3.interpolateInferno,
    plasma: d3.interpolatePlasma,
    viridis: d3.interpolateViridis,
    cividis: d3.interpolateCividis,
    turbo: d3.interpolateTurbo
  };
  return cmap[name] || d3.interpolateMagma;
}

export function drawHeatmapToCanvas(ctx, img, dstRect, { alpha = 0.65, colormap = d3.interpolateMagma } = {}) {
  // Draw heatmap image into an offscreen canvas, colormap it, then blit.
  const off = document.createElement("canvas");
  off.width = img.naturalWidth;
  off.height = img.naturalHeight;
  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.drawImage(img, 0, 0);

  const id = octx.getImageData(0, 0, off.width, off.height);
  const data = id.data;

  for (let i = 0; i < data.length; i += 4) {
    // assume heatmap is grayscale; use red channel
    const t = data[i] / 255;
    const c = d3.color(colormap(t));
    data[i] = c.r;
    data[i + 1] = c.g;
    data[i + 2] = c.b;
    data[i + 3] = Math.round(alpha * 255);
  }

  octx.putImageData(id, 0, 0);
  ctx.drawImage(off, dstRect.x, dstRect.y, dstRect.w, dstRect.h);
}
