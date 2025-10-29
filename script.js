// Convolution + Pooling Visualizer (no frameworks)
(function() {
  'use strict';

  // DOM elements
  const el = (id) => document.getElementById(id);
  const inputSizeEl = el('inputSize');
  const inputSizeValue = el('inputSizeValue');
  const kernelCountEl = el('kernelCount');
  const kernelCountValue = el('kernelCountValue');
  const inputCanvas = el('inputCanvas');
  const colKernels = el('colKernels');
  const colConvs = el('colConvs');
  const colPools = el('colPools');
  const colPooledStack = el('colPooledStack');

  // State (fixed architecture)
  const state = {
    channels: 3,           // RGB input
    inputSize: parseInt(inputSizeEl.value, 10),
    kernelCount: parseInt(kernelCountEl.value, 10),
    kernelSize: 3,         // 3x3 kernels
    convStride: 1,         // stride 1
    convPadding: 'valid',  // no padding
    poolType: 'max',       // max pool
    poolSize: 2,           // 2x2
    poolStride: 2          // stride 2
  };

  // Drawing helpers
  function drawOutlineGrid(canvas, h, w) {
    const padding = 8;
    const ctx = canvas.getContext('2d');
    const maxCell = 14;
    const parentWidth = canvas.parentElement ? canvas.parentElement.clientWidth - 2 : 320;
    const desiredMax = (canvas === inputCanvas) ? 300 : 260;
    const maxWidth = Math.min(parentWidth, desiredMax);
    const cell = Math.max(2, Math.min(maxCell, Math.floor((maxWidth - padding * 2) / w)));
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
  }

  function drawStackedGrid(canvas, h, w, c) {
    const padding = 8;
    const offset = 8;
    const ctx = canvas.getContext('2d');
    const maxCell = 14;
    const parentWidth = canvas.parentElement ? canvas.parentElement.clientWidth - 2 : 320;
    const desiredMax = (canvas === inputCanvas) ? 300 : 260;
    const maxWidth = Math.min(parentWidth, desiredMax);
    const cell = Math.max(2, Math.min(maxCell, Math.floor((maxWidth - padding * 2 - (c - 1) * offset) / w)));
    const width = cell * w + padding * 2 + (c - 1) * offset;
    const height = cell * h + padding * 2 + (c - 1) * offset;
    canvas.width = width; canvas.height = height;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    for (let layer = 0; layer < c; layer++) {
      const ox = padding + (c - 1 - layer) * offset;
      const oy = padding + (c - 1 - layer) * offset;
      ctx.strokeStyle = layer === c - 1 ? '#64748b' : '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.strokeRect(ox + 0.5, oy + 0.5, cell * w - 1, cell * h - 1);
    }
  }

  // Shapes
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

  // UI
  function syncLabels() {
    inputSizeValue.textContent = `${state.inputSize}×${state.inputSize}`;
    kernelCountValue.textContent = `${state.kernelCount}`;
  }

  function recomputeAndRender() {
    syncLabels();

    // Input (stacked RGB)
    drawStackedGrid(inputCanvas, state.inputSize, state.inputSize, state.channels);

    // Clear columns
    if (colKernels) colKernels.innerHTML = '';
    if (colConvs) colConvs.innerHTML = '';
    if (colPools) colPools.innerHTML = '';
    if (colPooledStack) colPooledStack.innerHTML = '';

    // Compute shapes once
    const [oh0, ow0] = computeConvOutputShape(state.inputSize, state.inputSize, state.kernelSize, state.convStride, state.convPadding);
    const convH = Math.max(1, oh0), convW = Math.max(1, ow0);
    const [ph0, pw0] = computePoolOutputShape(convH, convW, state.poolSize, state.poolStride);
    const poolH = Math.max(1, ph0), poolW = Math.max(1, pw0);

    // Per-kernel visuals
    for (let idx = 0; idx < state.kernelCount; idx++) {
      // Kernel (3D)
      const kCanvas = document.createElement('canvas');
      drawStackedGrid(kCanvas, state.kernelSize, state.kernelSize, state.channels);
      const kWrap = document.createElement('div'); kWrap.className = 'grid';
      const kLbl = document.createElement('div'); kLbl.className = 'label'; kLbl.textContent = 'Kernel weights (3D)';
      kWrap.appendChild(kCanvas); kWrap.appendChild(kLbl);
      colKernels.appendChild(kWrap);

      // Conv output (2D)
      const cCanvas = document.createElement('canvas');
      drawOutlineGrid(cCanvas, convH, convW);
      const cWrap = document.createElement('div'); cWrap.className = 'grid';
      const cLbl = document.createElement('div'); cLbl.className = 'label'; cLbl.textContent = 'Convolution output';
      cWrap.appendChild(cCanvas); cWrap.appendChild(cLbl);
      colConvs.appendChild(cWrap);

      // Pool output (2D)
      const pCanvas = document.createElement('canvas');
      drawOutlineGrid(pCanvas, poolH, poolW);
      const pWrap = document.createElement('div'); pWrap.className = 'grid';
      const pLbl = document.createElement('div'); pLbl.className = 'label'; pLbl.textContent = 'Pooled output';
      pWrap.appendChild(pCanvas); pWrap.appendChild(pLbl);
      colPools.appendChild(pWrap);
    }

    // Stacked pooled maps (depth = number of kernels)
    if (colPooledStack) {
      const sCanvas = document.createElement('canvas');
      drawStackedGrid(sCanvas, poolH, poolW, state.kernelCount);
      const sWrap = document.createElement('div'); sWrap.className = 'grid';
      const sLbl = document.createElement('div'); sLbl.className = 'label'; sLbl.textContent = 'Stacked pooled maps';
      sWrap.appendChild(sCanvas); sWrap.appendChild(sLbl);
      colPooledStack.appendChild(sWrap);
    }
  }

  function wire() {
    inputSizeEl.addEventListener('input', () => {
      state.inputSize = parseInt(inputSizeEl.value, 10);
      recomputeAndRender();
    });
    kernelCountEl.addEventListener('input', () => {
      state.kernelCount = parseInt(kernelCountEl.value, 10);
      recomputeAndRender();
    });
    window.addEventListener('resize', recomputeAndRender);
  }

  // Init
  wire();
  recomputeAndRender();
})();
