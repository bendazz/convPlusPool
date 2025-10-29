// Convolution + Pooling Visualizer (no frameworks)
(function() {
  'use strict';

  // DOM elements
  const el = (id) => document.getElementById(id);
  const inputSizeEl = el('inputSize');
  const inputSizeValue = el('inputSizeValue');
  const kernelCountEl = el('kernelCount');
  const kernelCountValue = el('kernelCountValue');
  // pool size fixed; no DOM inputs
  const inputCanvas = el('inputCanvas');
  const kernelsContainer = el('kernelsContainer');

  // State
  let state = {
    channels: 3, // RGB input
    inputSize: parseInt(inputSizeEl.value, 10),
    kernelCount: parseInt(kernelCountEl.value, 10),
  kernelSize: 3, // fixed 3x3
    convStride: 1, // fixed
    convPadding: 'valid', // fixed (no padding)
  poolType: 'max', // fixed
  poolSize: 2, // fixed 2x2
    poolStride: 2 // fixed
  };

  // Utilities
  function seededRandom(seed) {
    let x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }

  function randn() {
    // Box-Muller transform for a simple normal-ish distribution
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function zeros(h, w) {
    return Array.from({ length: h }, () => Array(w).fill(0));
  }

  function randomGrid(h, w) {
    const g = zeros(h, w);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        // bounded random [0,1]
        g[i][j] = Math.min(1, Math.max(0, 0.5 + 0.25 * randn()));
      }
    }
    return g;
  }

  function randomKernel(k) {
    const g = zeros(k, k);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        g[i][j] = randn();
      }
    }
    // normalize to zero mean, unit-ish variance
    const flat = g.flat();
    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    const centered = flat.map(v => v - mean);
    const std = Math.sqrt(centered.reduce((a, b) => a + b * b, 0) / centered.length) || 1;
    const it = centered.map(v => v / std);
    let idx = 0;
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        g[i][j] = it[idx++];
      }
    }
    return g;
  }

  function pad2D(arr, pad) {
    const h = arr.length, w = arr[0].length;
    const out = zeros(h + 2 * pad, w + 2 * pad);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        out[i + pad][j + pad] = arr[i][j];
      }
    }
    return out;
  }

  function convolve2D(input, kernel, stride = 1, padding = 'valid') {
    const ih = input.length, iw = input[0].length;
    const kh = kernel.length, kw = kernel[0].length;

    let x = input;
    let pad = 0;
    if (padding === 'same') {
      // same padding for stride 1 approximated: pad so output size ~ input size
      const padH = Math.floor((kh - 1) / 2);
      const padW = Math.floor((kw - 1) / 2);
      if (padH !== padW) {
        // Use symmetric single-value pad for simplicity
      }
      pad = padH; // assume square kernel
      x = pad2D(input, pad);
    }

    const oh = Math.floor((x.length - kh) / stride) + 1;
    const ow = Math.floor((x[0].length - kw) / stride) + 1;
    const out = zeros(oh, ow);

    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let sum = 0;
        for (let ki = 0; ki < kh; ki++) {
          for (let kj = 0; kj < kw; kj++) {
            sum += x[i * stride + ki][j * stride + kj] * kernel[ki][kj];
          }
        }
        out[i][j] = sum;
      }
    }
    return out;
  }

  function pool2D(input, size = 2, stride = 2, type = 'max') {
    const ih = input.length, iw = input[0].length;
    const oh = Math.floor((ih - size) / stride) + 1;
    const ow = Math.floor((iw - size) / stride) + 1;
    const out = zeros(oh, ow);

    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let v;
        if (type === 'max') v = -Infinity; else v = 0;
        for (let pi = 0; pi < size; pi++) {
          for (let pj = 0; pj < size; pj++) {
            const val = input[i * stride + pi][j * stride + pj];
            if (type === 'max') v = Math.max(v, val);
            else v += val;
          }
        }
        out[i][j] = type === 'max' ? v : v / (size * size);
      }
    }
    return out;
  }

  // Color mapping (blue->white->red) for symmetric values, and white->orange for [0,1]
  function colorFor(value, min, max) {
    if (min >= 0) {
      // [0,1] like inputs: white (0) to orange (1)
      const t = (value - min) / (max - min + 1e-8);
      const h = 35; // orange
      const s = 90;
      const l = 100 - 60 * t; // from near-white to richer color
      return `hsl(${h} ${s}% ${l}%)`;
    } else {
      // symmetric for conv outputs: blue (neg) -> white -> red (pos)
      const a = Math.max(Math.abs(min), Math.abs(max)) + 1e-8;
      const t = value / a; // -1..1
      if (t >= 0) {
        // white to red
        const h = 0; // red
        const s = 80;
        const l = 100 - 60 * t;
        return `hsl(${h} ${s}% ${l}%)`;
      } else {
        // white to blue
        const h = 220; // blue
        const s = 80;
        const l = 100 - 60 * (-t);
        return `hsl(${h} ${s}% ${l}%)`;
      }
    }
  }

  function drawGrid(canvas, grid, opts = {}) {
    const showNumbers = opts.showNumbers ?? true;
    const padding = 8;
    const h = grid.length, w = grid[0].length;

    const ctx = canvas.getContext('2d');

    // Resize canvas to fit grid
    const maxCell = 28;
    const maxWidth = canvas.parentElement ? canvas.parentElement.clientWidth - 2 : 320;
    const cell = Math.min(maxCell, Math.floor((maxWidth - padding * 2) / w));
    const width = cell * w + padding * 2;
    const height = cell * h + padding * 2;
    canvas.width = width; canvas.height = height;

    const flat = grid.flat();
    let min = Math.min(...flat), max = Math.max(...flat);
    if (min === max) { min -= 1; max += 1; }

    // background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // cells
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `${Math.max(10, Math.floor(cell * 0.38))}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace`;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const x = padding + j * cell;
        const y = padding + i * cell;
        const v = grid[i][j];
        ctx.fillStyle = colorFor(v, min, max);
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = '#e5e7eb';
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        if (showNumbers && cell >= 16) {
          ctx.fillStyle = '#111827';
          const text = (Math.abs(v) < 1e-4 ? 0 : v).toFixed(2);
          ctx.fillText(text, x + cell / 2, y + cell / 2);
        }
      }
    }
  }

  // Outline-only grid drawing for diagram mode (single 2D grid)
  function drawOutlineGrid(canvas, h, w, label) {
    const padding = 8;
    const ctx = canvas.getContext('2d');
    const maxCell = 28;
    const maxWidth = canvas.parentElement ? canvas.parentElement.clientWidth - 2 : 320;
    const cell = Math.min(maxCell, Math.floor((maxWidth - padding * 2) / w));
    const width = cell * w + padding * 2;
    const height = cell * h + padding * 2;
    canvas.width = width; canvas.height = height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#cbd5e1';
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const x = padding + j * cell;
        const y = padding + i * cell;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }

    // Label in the corner
    ctx.fillStyle = '#334155';
    ctx.font = `${Math.max(10, Math.floor(cell * 0.38))}px ui-sans-serif, system-ui, -apple-system`;
    const text = label || `${h}×${w}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(text, width - 6, height - 4);
  }

  // Stacked grids to represent channels depth (C)
  function drawStackedGrid(canvas, h, w, c, label) {
    const padding = 12;
    const offset = 8; // per-layer offset (px)
    const ctx = canvas.getContext('2d');
    const maxCell = 24;
    const maxWidth = canvas.parentElement ? canvas.parentElement.clientWidth - 2 : 320;
    // Account for stack offset in width
    const cell = Math.min(maxCell, Math.floor((maxWidth - padding * 2 - (c - 1) * offset) / w));
    const width = cell * w + padding * 2 + (c - 1) * offset;
    const height = cell * h + padding * 2 + (c - 1) * offset;
    canvas.width = width; canvas.height = height;

    // background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw back to front (no per-cell gridlines; just layer borders)
    for (let layer = 0; layer < c; layer++) {
      const ox = padding + (c - 1 - layer) * offset;
      const oy = padding + (c - 1 - layer) * offset;
      // Outer border for each layer box
      ctx.strokeStyle = layer === c - 1 ? '#64748b' : '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.strokeRect(ox + 0.5, oy + 0.5, cell * w - 1, cell * h - 1);
    }

    // Label on the front-most layer
    ctx.fillStyle = '#334155';
    ctx.font = `${Math.max(10, Math.floor(cell * 0.38))}px ui-sans-serif, system-ui, -apple-system`;
    const text = label || `${h}×${w}×${c}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(text, width - 6, height - 4);
  }

  function computeConvOutputShape(inH, inW, k, stride, padding) {
    let pad = 0;
    if (padding === 'same') pad = Math.floor((k - 1) / 2);
    const ph = inH + 2 * pad;
    const pw = inW + 2 * pad;
    const oh = Math.floor((ph - k) / stride) + 1;
    const ow = Math.floor((pw - k) / stride) + 1;
    return [Math.max(0, oh), Math.max(0, ow)];
  }

  function computePoolOutputShape(inH, inW, size, stride) {
    const oh = Math.floor((inH - size) / stride) + 1;
    const ow = Math.floor((inW - size) / stride) + 1;
    return [Math.max(0, oh), Math.max(0, ow)];
  }

  function makeKernelCard(index) {
    const card = document.createElement('div');
    card.className = 'kernel-card';

    const header = document.createElement('div');
    header.className = 'kernel-header';
    const title = document.createElement('h3');
    title.textContent = `Kernel ${index + 1}`;
    title.style.margin = '0';
    const tag = document.createElement('span');
    tag.className = 'tag';
  tag.textContent = `${state.kernelSize}×${state.kernelSize}×${state.channels}`;
    header.appendChild(title);
    header.appendChild(tag);

    const grids = document.createElement('div');
    grids.className = 'kernel-grids';

    const kernelDiv = document.createElement('div');
    kernelDiv.className = 'grid';
    const kernelCanvas = document.createElement('canvas');
    const kernelLabel = document.createElement('div');
    kernelLabel.className = 'label';
  kernelLabel.textContent = 'Kernel weights (3D)';
    kernelDiv.appendChild(kernelCanvas);
    kernelDiv.appendChild(kernelLabel);

    const convDiv = document.createElement('div');
    convDiv.className = 'grid';
    const convCanvas = document.createElement('canvas');
    const convLabel = document.createElement('div');
    convLabel.className = 'label';
    convLabel.textContent = 'Convolution output';
    convDiv.appendChild(convCanvas);
    convDiv.appendChild(convLabel);

    const poolDiv = document.createElement('div');
    poolDiv.className = 'grid';
    const poolCanvas = document.createElement('canvas');
    const poolLabel = document.createElement('div');
    poolLabel.className = 'label';
    poolLabel.textContent = 'Pooled output';
    poolDiv.appendChild(poolCanvas);
    poolDiv.appendChild(poolLabel);

    grids.appendChild(kernelDiv);
    grids.appendChild(convDiv);
    grids.appendChild(poolDiv);

    card.appendChild(header);
    card.appendChild(grids);

    return { card, kernelCanvas, convCanvas, poolCanvas };
  }

  function syncLabels() {
    inputSizeValue.textContent = `${state.inputSize}×${state.inputSize}`;
    kernelCountValue.textContent = `${state.kernelCount}`;
  // kernel size fixed; no dynamic label element
  // pool size fixed; no dynamic label element
  }

  function recomputeAndRender() {
    syncLabels();

    // draw input (diagram only)
    drawStackedGrid(inputCanvas, state.inputSize, state.inputSize, state.channels, `Input ${state.inputSize}×${state.inputSize}×${state.channels}`);

    // rebuild kernel cards
    kernelsContainer.innerHTML = '';

    for (let idx = 0; idx < state.kernelCount; idx++) {
      const { card, kernelCanvas, convCanvas, poolCanvas } = makeKernelCard(idx);
      kernelsContainer.appendChild(card);

      // Diagram path: outline only with dimensions
    drawStackedGrid(kernelCanvas, state.kernelSize, state.kernelSize, state.channels, `K ${state.kernelSize}×${state.kernelSize}×${state.channels}`);
      const [oh, ow] = computeConvOutputShape(state.inputSize, state.inputSize, state.kernelSize, state.convStride, state.convPadding);
      const convLabel = oh > 0 && ow > 0 ? `Conv ${oh}×${ow}` : 'Conv n/a';
      drawOutlineGrid(convCanvas, Math.max(1, oh), Math.max(1, ow), convLabel);
      const [ph, pw] = computePoolOutputShape(Math.max(1, oh), Math.max(1, ow), state.poolSize, state.poolStride);
      const poolLabel = (oh >= state.poolSize && ow >= state.poolSize) && ph > 0 && pw > 0 ? `Pool ${ph}×${pw}` : 'Pool n/a';
      drawOutlineGrid(poolCanvas, Math.max(1, ph), Math.max(1, pw), poolLabel);
    }
  }

  // Event wiring
  function wire() {
    inputSizeEl.addEventListener('input', () => {
      state.inputSize = parseInt(inputSizeEl.value, 10);
      recomputeAndRender();
    });

    kernelCountEl.addEventListener('input', () => {
      state.kernelCount = parseInt(kernelCountEl.value, 10);
      recomputeAndRender();
    });

    // kernel size fixed: no listener

    // pool type fixed: no listener

    // pool size fixed: no listener

    // Resize redraw
    window.addEventListener('resize', () => {
      recomputeAndRender();
    });
  }

  // Init
  function init() {
    wire();
    recomputeAndRender();
  }

  init();
})();
