/* --- EXACT STITCH REPLICA MULTI-PAGE CONTROLLER --- */

document.addEventListener('DOMContentLoaded', () => {
  
  // PAGE NAVIGATION ROUTER
  const navHome = document.getElementById('nav-btn-home');
  const navAnalyze = document.getElementById('nav-btn-analyze');
  const navResearch = document.getElementById('nav-btn-research');
  const heroStartBtn = document.getElementById('hero-start-btn');
  const logoClick = document.getElementById('logo-click');

  function navigateTo(pageId) {
    document.body.setAttribute('data-page', pageId);
    
    // Clear active header links
    [navHome, navAnalyze, navResearch].forEach(btn => btn.classList.remove('active'));
    
    // Highlight correct link
    if (pageId === 'home') {
      navHome.classList.add('active');
    } else if (pageId === 'analyze') {
      navAnalyze.classList.add('active');
    } else if (pageId === 'research') {
      navResearch.classList.add('active');
    }

    // Reset SVG paths inside charts for animation trigger
    const chartPaths = document.querySelectorAll('.chart-path-anim');
    chartPaths.forEach(path => {
      path.style.animation = 'none';
      path.offsetHeight; /* trigger reflow */
      path.style.animation = null;
    });
  }

  // Bind clicks
  logoClick.addEventListener('click', (e) => { e.preventDefault(); navigateTo('home'); });
  navHome.addEventListener('click', (e) => { e.preventDefault(); navigateTo('home'); });
  navAnalyze.addEventListener('click', (e) => { e.preventDefault(); navigateTo('analyze'); });
  navResearch.addEventListener('click', (e) => { e.preventDefault(); navigateTo('research'); });
  heroStartBtn.addEventListener('click', () => navigateTo('analyze'));

  // --- SCREEN 2: ANALYZE DASHBOARD UPLOADS ---
  const API_URL = 'http://127.0.0.1:5000/analyze';
  const API_PREDICT_URL = 'http://127.0.0.1:5000/predict';
  const dragDropArea = document.getElementById('drag-drop-area');
  const dashboardFileUpload = document.getElementById('dashboard-file-upload');

  const imgOriginal = document.getElementById('img-original');
  const imgPointer = document.getElementById('img-pointer');
  const imgHeatmapBase = document.getElementById('img-heatmap-base');
  const imgGradcamBase = document.getElementById('img-gradcam-base');
  const imgMiniHeatmap = document.getElementById('img-mini-heatmap');
  const imgMiniGradcam = document.getElementById('img-mini-gradcam');

  const healthStatusSingle = document.getElementById('health-status-single');

  const insightClassification = document.getElementById('insight-classification');
  const insightDesc = document.getElementById('insight-desc');
  const confidenceBar = document.getElementById('telemetry-confidence-bar');
  const confidenceVal = document.getElementById('telemetry-confidence-val');
  const probabilityVal = document.getElementById('telemetry-probability');
  const latencyVal = document.getElementById('telemetry-latency');
  const debugTopPrediction = document.getElementById('debug-top-prediction');
  const debugRawOutputs = document.getElementById('debug-raw-outputs');
  const debugClassOrder = document.getElementById('debug-class-order');
  const debugSpectrumMeta = document.getElementById('debug-spectrum-meta');
  const spectrumHealthy = document.getElementById('spectrum-healthy');
  const spectrumRotten = document.getElementById('spectrum-rotten');
  const spectrumUnknown = document.getElementById('spectrum-unknown');
  const spectrumHealthyVal = document.getElementById('spectrum-healthy-val');
  const spectrumRottenVal = document.getElementById('spectrum-rotten-val');
  const spectrumUnknownVal = document.getElementById('spectrum-unknown-val');
  const metricAccuracyValue = document.getElementById('metric-accuracy-value');
  const metricLossValue = document.getElementById('metric-loss-value');

  const statusConfig = {
    healthy: {
      label: 'Healthy, Grade A.',
      color: '#4caf50',
      fillGradient: 'linear-gradient(90deg, #4caf50, #81c784)',
      badgeClass: 'healthy',
      theme: ['#effaf1', '#dcf8e6', '#e7f7eb', '#dff4e3']
    },
    rotten: {
      label: 'Rotten / Fungal Anthracnose.',
      color: '#f44336',
      fillGradient: 'linear-gradient(90deg, #f44336, #e57373)',
      badgeClass: 'rotten',
      theme: ['#fff0f0', '#ffdcdc', '#ffe8d8', '#ffd0cf']
    },
    unknown: {
      label: 'Uncertain Specimen / Out of Distribution.',
      color: '#ff9800',
      fillGradient: 'linear-gradient(90deg, #ff9800, #ffb74d)',
      badgeClass: 'unknown',
      theme: ['#fff8e6', '#fff0c9', '#fff1d6', '#ffe8a7']
    }
  };

  const transparentPixel = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
  let lastUploadedFile = null;
  let lastPointerPoint = { x: 50, y: 50 };
  let focusDebounceTimer = null;
  let focusRequestSeq = 0;

  function setAnalyzeTheme(statusKey) {
    const config = statusConfig[statusKey] || statusConfig.unknown;
    const theme = config.theme;
    document.body.style.background = `linear-gradient(-45deg, ${theme[0]}, ${theme[1]}, ${theme[2]}, ${theme[3]})`;
    document.body.style.backgroundSize = '400% 400%';
    document.body.style.animation = 'waveGradient 15s ease infinite';
  }

  function setImageSources(sourceUrl) {
    document.querySelectorAll('.card-image-box, .mini-img-wrap').forEach((box) => {
      box.style.backgroundImage = `url("${sourceUrl}")`;
      box.style.backgroundRepeat = 'no-repeat';
      box.style.backgroundPosition = 'center';
      box.style.backgroundSize = 'cover';
    });
  }

  function setOverlayImage(maskElement, imageUrl, blendMode = 'normal') {
    if (!maskElement) {
      return;
    }
    // Set the overlay image with proper styling
    // Fade out, swap image, fade in for a smooth transition
    try {
      maskElement.style.transition = maskElement.style.transition || 'opacity 220ms ease, transform 220ms ease';
      maskElement.style.opacity = '0';
      // Allow paint to clear before changing background
      requestAnimationFrame(() => {
        maskElement.style.backgroundColor = 'transparent';
        maskElement.style.backgroundImage = `url("${imageUrl}")`;
        maskElement.style.backgroundRepeat = 'no-repeat';
        maskElement.style.backgroundPosition = 'center';
        maskElement.style.backgroundSize = 'cover';
        maskElement.style.filter = 'saturate(1.2) contrast(1.1) brightness(1.05)';
        maskElement.style.mixBlendMode = blendMode || 'normal';
        // small delay to ensure the new background is painted then fade in
        setTimeout(() => { maskElement.style.opacity = '0.85'; }, 30);
      });
    } catch (e) {
      // fallback to immediate set
      maskElement.style.backgroundColor = 'transparent';
      maskElement.style.backgroundImage = `url("${imageUrl}")`;
      maskElement.style.backgroundRepeat = 'no-repeat';
      maskElement.style.backgroundPosition = 'center';
      maskElement.style.backgroundSize = 'cover';
      maskElement.style.opacity = '0.85';
      try { maskElement.style.mixBlendMode = blendMode || 'normal'; } catch (err) {}
    }
  }

  function ensureHeatmapLegend() {
    const container = document.querySelector('.insight-glass-card');
    if (!container) return null;
    let legend = container.querySelector('.heatmap-legend');
    if (legend) return legend;
    legend = document.createElement('div');
    legend.className = 'heatmap-legend';
    legend.innerHTML = `
      <div class="legend-title">Explanation</div>
      <div class="legend-line">Hotspot colors indicate model attention (warmer = higher).</div>
      <div class="legend-scale">
        <span class="legend-stop" style="background:linear-gradient(90deg,#00f3ff,#ffeb3b,#f44336)"></span>
        <span class="legend-label">Low</span>
        <span class="legend-label" style="margin-left:8px">High</span>
      </div>
      <div class="legend-meta">Layer: <span id="legend-used-layer">n/a</span> &nbsp; Peak: <span id="legend-heatmap-peak">0.0</span></div>
      <div class="legend-note">Grad-CAM shows convolutional attention; Input-gradient fallback shows saliency.</div>
    `;
    container.appendChild(legend);
    return legend;
  }

  function resetOverlayImage(maskElement) {
    if (!maskElement) {
      return;
    }

    maskElement.style.background = '';
    maskElement.style.backgroundImage = '';
    maskElement.style.backgroundRepeat = '';
    maskElement.style.backgroundPosition = '';
    maskElement.style.backgroundSize = '';
    maskElement.style.mixBlendMode = '';
    maskElement.style.opacity = '';
  }

  function clearPointerLayer(pointerBox) {
    if (!pointerBox) return;
    const existingLayer = pointerBox.querySelector('.pointer-overlay-layer');
    if (existingLayer) {
      existingLayer.remove();
    }
  }

  function installDraggablePointer(pointerBox, focusPoints = [], onChange = null, initialPoint = null) {
    if (!pointerBox) return;

    clearPointerLayer(pointerBox);

    const layer = document.createElement('div');
    layer.className = 'pointer-overlay-layer';

    const pointer = document.createElement('div');
    pointer.className = 'pointer-draggable';
    pointer.setAttribute('aria-label', 'Draggable pointer');
    pointer.setAttribute('role', 'img');

    const pointSeed = initialPoint || (Array.isArray(focusPoints) && focusPoints.length > 0 ? focusPoints[0] : { x: 50, y: 50 });
    let posX = Number(pointSeed.x) || 50;
    let posY = Number(pointSeed.y) || 50;

    const applyPosition = () => {
      pointer.style.left = `${Math.max(0, Math.min(100, posX))}%`;
      pointer.style.top = `${Math.max(0, Math.min(100, posY))}%`;
    };

    const emitChange = () => {
      if (typeof onChange === 'function') {
        onChange({ x: posX, y: posY });
      }
    };

    let dragging = false;
    let pointerId = null;

    const updateFromEvent = (event) => {
      const rect = pointerBox.getBoundingClientRect();
      const clientX = event.clientX ?? (event.touches && event.touches[0] ? event.touches[0].clientX : null);
      const clientY = event.clientY ?? (event.touches && event.touches[0] ? event.touches[0].clientY : null);
      if (clientX == null || clientY == null) return;

      const x = ((clientX - rect.left) / rect.width) * 100;
      const y = ((clientY - rect.top) / rect.height) * 100;
      posX = Math.max(0, Math.min(100, x));
      posY = Math.max(0, Math.min(100, y));
      applyPosition();
      emitChange();
    };

    const onPointerMove = (event) => {
      if (!dragging || (pointerId != null && event.pointerId !== pointerId)) return;
      event.preventDefault();
      updateFromEvent(event);
    };

    const onPointerUp = (event) => {
      if (pointerId != null && event.pointerId !== pointerId) return;
      dragging = false;
      pointerId = null;
      try { pointer.releasePointerCapture(event.pointerId); } catch (e) {}
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
    };

    pointer.addEventListener('pointerdown', (event) => {
      dragging = true;
      pointerId = event.pointerId;
      event.preventDefault();
      try { pointer.setPointerCapture(event.pointerId); } catch (e) {}
      updateFromEvent(event);
      window.addEventListener('pointermove', onPointerMove, { passive: false });
      window.addEventListener('pointerup', onPointerUp, { passive: false });
      window.addEventListener('pointercancel', onPointerUp, { passive: false });
    });

    applyPosition();
    emitChange();
    layer.appendChild(pointer);
    pointerBox.appendChild(layer);
  }

  function renderPointerPanel(pointerBox, baseImageUrl, focusPoints = []) {
    if (!pointerBox) return;
    pointerBox.style.backgroundImage = `url("${baseImageUrl}")`;
    pointerBox.style.backgroundSize = 'cover';
    pointerBox.style.backgroundPosition = 'center';
    pointerBox.style.backgroundRepeat = 'no-repeat';
    installDraggablePointer(pointerBox, focusPoints, scheduleFocusedPrediction, lastPointerPoint);
  }

  function applyFocusedResponse(data) {
    setOverlayImage(document.querySelector('.heatmap-overlay-box .heatmap-color-mask'), data.heatmap_overlay || transparentPixel, 'normal');
    setOverlayImage(document.querySelector('.gradcam-overlay-box .gradcam-color-mask'), data.gradcam_overlay || transparentPixel, 'normal');
    data.class_names = data.class_names || ['healthy', 'rotten', 'unknown'];
    setStatusState(data.prediction || 'unknown', data);
  }

  async function requestFocusedPrediction(point) {
    if (!lastUploadedFile) return;
    const requestId = ++focusRequestSeq;
    try {
      const formData = new FormData();
      formData.append('file', lastUploadedFile);
      formData.append('focus_x', String(Number(point?.x) || 50));
      formData.append('focus_y', String(Number(point?.y) || 50));
      // Slightly smaller crop yields faster, more localized updates
      formData.append('crop_pct', '45');

      const response = await fetchWithRetry(API_PREDICT_URL, { method: 'POST', body: formData }, 2);

      const data = await response.json();
      if (requestId !== focusRequestSeq) return;
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Focused inference failed.');
      }
      applyFocusedResponse(data);
    } catch (error) {
      console.error('Focused analysis error:', error);
    }
  }

  async function fetchWithRetry(url, opts, retries = 2, backoff = 300) {
    let lastErr = null;
    for (let i = 0; i <= retries; i++) {
      try {
        const res = await fetch(url, opts);
        if (!res.ok) {
          // Attempt to parse server JSON error for better message
          let body = null;
          try { body = await res.json(); } catch (e) { /* ignore */ }
          const msg = body && body.error ? body.error : `HTTP ${res.status}`;
          throw new Error(msg);
        }
        return res;
      } catch (err) {
        lastErr = err;
        if (i < retries) {
          await new Promise(r => setTimeout(r, backoff * (i + 1)));
          continue;
        }
        throw lastErr;
      }
    }
    throw lastErr;
  }

  function scheduleFocusedPrediction(point) {
    lastPointerPoint = {
      x: Number(point?.x) || 50,
      y: Number(point?.y) || 50,
    };
    if (focusDebounceTimer) {
      clearTimeout(focusDebounceTimer);
    }
    // Reduce debounce for snappier UI while still throttling rapid pointer moves
    focusDebounceTimer = setTimeout(() => {
      requestFocusedPrediction(lastPointerPoint);
    }, 120);
  }

  // Pointer rendering removed - pointers are now drawn on images in Python backend

  function setStatusState(statusKey, result) {
    const config = statusConfig[statusKey] || statusConfig.unknown;
    setAnalyzeTheme(statusKey);
    const probability = Number(result?.probability) || 0;
    const confidencePercent = Number(result?.confidence) || 0;
    const latency = Number(result?.latency_ms) || 0;

    if (healthStatusSingle) {
      const confText = Number.isFinite(confidencePercent) ? `${confidencePercent.toFixed(2)}%` : '--';
      const simpleLabel = (statusKey === 'healthy') ? 'Healthy' : (statusKey === 'rotten') ? 'Rotten' : 'Unknown';
      healthStatusSingle.innerHTML = `${simpleLabel} <span class="status-confidence">${confText}</span>`;
      healthStatusSingle.style.background = config.fillGradient;
      healthStatusSingle.style.color = '#ffffff';
      healthStatusSingle.title = `${simpleLabel} — confidence ${confText}`;
    }
    const reason = result?.decision_reason || 'No explanation returned by the backend.';
    const probabilities = result?.probabilities || {};
    const rawVector = Array.isArray(result?.raw_outputs) ? result.raw_outputs.map((value) => Number(value) || 0) : [];
    const displayProbability = Number.isFinite(probability) ? probability.toFixed(4) : '0.0000';
    const displayConfidence = Number.isFinite(confidencePercent) ? `${confidencePercent.toFixed(2)}%` : '0.00%';
    const vectorTotal = rawVector.reduce((sum, value) => sum + Math.max(value, 0), 0) || 1;
    const vectorHealthy = probabilities.healthy ?? (rawVector[0] ?? 0);
    const vectorRotten = probabilities.rotten ?? (rawVector[1] ?? 0);
    const vectorUnknown = probabilities.unknown ?? (rawVector[2] ?? 0);
    const vectorEntropy = [vectorHealthy, vectorRotten, vectorUnknown]
      .filter((value) => Number.isFinite(value) && value > 0)
      .reduce((sum, value) => sum - value * Math.log(value), 0);
    const normalisedEntropy = Number.isFinite(vectorEntropy) ? (vectorEntropy / Math.log(3)).toFixed(3) : '0.000';
    const topTwoSorted = [vectorHealthy, vectorRotten, vectorUnknown].filter((value) => Number.isFinite(value)).sort((a, b) => b - a);
    const topMargin = topTwoSorted.length > 1 ? (topTwoSorted[0] - topTwoSorted[1]).toFixed(4) : '0.0000';

    if (insightClassification) {
      insightClassification.textContent = config.label;
      insightClassification.style.color = config.color;
      insightClassification.className = 'insight-value-highlight';
    }

    if (insightDesc) {
      const healthyScore = probabilities.healthy != null ? `Healthy ${Math.round(probabilities.healthy * 100)}%` : 'Healthy n/a';
      const rottenScore = probabilities.rotten != null ? `Rotten ${Math.round(probabilities.rotten * 100)}%` : 'Rotten n/a';
      const unknownScore = probabilities.unknown != null ? `Unknown ${Math.round(probabilities.unknown * 100)}%` : 'Unknown n/a';
      insightDesc.innerHTML = `<span class="insight-label-bold">Why:</span> ${reason} ${healthyScore}. ${rottenScore}. ${unknownScore}.`;
    }

    if (confidenceBar) {
      confidenceBar.style.width = displayConfidence;
      confidenceBar.style.background = config.fillGradient;
    }
    if (confidenceVal) {
      confidenceVal.textContent = displayConfidence;
      confidenceVal.style.color = config.color;
    }
    if (probabilityVal) {
      probabilityVal.textContent = displayProbability;
    }
    if (latencyVal) {
      latencyVal.textContent = `${latency.toFixed(1)} ms`;
    }

    if (debugTopPrediction) {
      debugTopPrediction.textContent = result?.raw_prediction || statusKey;
    }
    if (debugRawOutputs) {
      const rawOutputs = Array.isArray(result?.raw_outputs) ? result.raw_outputs : [];
      debugRawOutputs.textContent = rawOutputs.length > 0 ? rawOutputs.map((value) => Number(value).toFixed(4)).join(', ') : 'n/a';
    }
    if (debugClassOrder) {
      debugClassOrder.textContent = Array.isArray(result?.class_names) ? result.class_names.join(' / ') : 'healthy / rotten / unknown';
    }
    const debugUsedLayer = document.getElementById('debug-used-layer');
    const debugHeatmapPeak = document.getElementById('debug-heatmap-peak');
    if (debugUsedLayer) debugUsedLayer.textContent = result?.used_layer || result?.model_info?.last_conv_layer || 'n/a';
    if (debugHeatmapPeak) debugHeatmapPeak.textContent = (result?.heatmap_peak != null) ? String(result.heatmap_peak) : '0.0';

      // Also update legend meta if present
      const legendLayer = document.getElementById('legend-used-layer');
      const legendPeak = document.getElementById('legend-heatmap-peak');
      if (legendLayer) legendLayer.textContent = result?.used_layer || result?.model_info?.last_conv_layer || 'n/a';
      if (legendPeak) legendPeak.textContent = (result?.heatmap_peak != null) ? String(result.heatmap_peak) : '0.0';

    if (debugSpectrumMeta) {
      debugSpectrumMeta.textContent = `margin ${topMargin} | entropy ${normalisedEntropy}`;
    }
    if (spectrumHealthy) {
      spectrumHealthy.style.width = `${Math.max(0, Math.min(100, vectorHealthy * 100))}%`;
    }
    if (spectrumRotten) {
      spectrumRotten.style.width = `${Math.max(0, Math.min(100, vectorRotten * 100))}%`;
    }
    if (spectrumUnknown) {
      spectrumUnknown.style.width = `${Math.max(0, Math.min(100, vectorUnknown * 100))}%`;
    }
    if (spectrumHealthyVal) {
      spectrumHealthyVal.textContent = Number.isFinite(vectorHealthy) ? vectorHealthy.toFixed(4) : '0.0000';
    }
    if (spectrumRottenVal) {
      spectrumRottenVal.textContent = Number.isFinite(vectorRotten) ? vectorRotten.toFixed(4) : '0.0000';
    }
    if (spectrumUnknownVal) {
      spectrumUnknownVal.textContent = Number.isFinite(vectorUnknown) ? vectorUnknown.toFixed(4) : '0.0000';
    }

    updateChartTheme(statusKey);
  }

  function setLoadingState() {
    const badgeNodes = [document.getElementById('badge-healthy'), document.getElementById('badge-rotten'), document.getElementById('badge-unknown')].filter(Boolean);
    badgeNodes.forEach((badge) => badge.classList.remove('active'));
    setAnalyzeTheme('unknown');
    if (insightClassification) {
      insightClassification.textContent = 'Analyzing...';
      insightClassification.style.color = '#ffffff';
    }
    if (insightDesc) {
      insightDesc.innerHTML = '<span class="insight-label-bold">Why:</span> The backend is processing the image and generating the explanation maps.';
    }
    if (confidenceBar) {
      confidenceBar.style.width = '12%';
      confidenceBar.style.background = 'linear-gradient(90deg, #64748b, #94a3b8)';
    }
    if (confidenceVal) {
      confidenceVal.textContent = '--';
      confidenceVal.style.color = '#ffffff';
    }
    if (probabilityVal) {
      probabilityVal.textContent = '--';
    }
    if (latencyVal) {
      latencyVal.textContent = '--';
    }
    if (debugTopPrediction) {
      debugTopPrediction.textContent = '--';
    }
    if (debugRawOutputs) {
      debugRawOutputs.textContent = '--';
    }
    if (debugClassOrder) {
      debugClassOrder.textContent = 'healthy / rotten / unknown';
    }
    if (debugSpectrumMeta) {
      debugSpectrumMeta.textContent = 'p1 / p2 / p3';
    }
    if (spectrumHealthy) {
      spectrumHealthy.style.width = '0%';
    }
    if (spectrumRotten) {
      spectrumRotten.style.width = '0%';
    }
    if (spectrumUnknown) {
      spectrumUnknown.style.width = '0%';
    }
    if (spectrumHealthyVal) {
      spectrumHealthyVal.textContent = '0.0000';
    }
    if (spectrumRottenVal) {
      spectrumRottenVal.textContent = '0.0000';
    }
    if (spectrumUnknownVal) {
      spectrumUnknownVal.textContent = '0.0000';
    }

    updateChartTheme('unknown');
  }

  // -------- Runtime config helpers (talk to /config and /reload_model) --------
  const settingsPreprocess = document.getElementById('settings-preprocess');
  const settingsForce = document.getElementById('settings-force-classification');
  const settingsMinConfidence = document.getElementById('settings-min-confidence');
  const settingsMinMargin = document.getElementById('settings-min-margin');
  const settingsLastConv = document.getElementById('settings-last-conv');
  const btnApplyConfig = document.getElementById('btn-apply-config');
  const btnReloadModel = document.getElementById('btn-reload-model');
  const btnQuickTest = document.getElementById('btn-quick-test');
  const configStatus = document.getElementById('config-status');

  async function fetchRuntimeConfig() {
    try {
      const res = await fetch('http://127.0.0.1:5000/config');
      if (!res.ok) throw new Error('Failed to fetch config');
      const json = await res.json();
      settingsPreprocess.value = json.preprocess_mode || 'auto';
      settingsForce.checked = !!json.force_classification;
      settingsMinConfidence.value = Number.isFinite(json.min_confidence) ? json.min_confidence : 0.45;
      settingsMinMargin.value = Number.isFinite(json.min_margin) ? json.min_margin : 0.08;
      settingsLastConv.value = json.last_conv_layer_override || '';
      configStatus.textContent = json.mock_inference ? 'MOCK mode' : '';
      const mockBanner = document.getElementById('mock-mode-banner');
      if (mockBanner) {
        if (json.mock_inference) {
          mockBanner.style.display = 'block';
        } else {
          mockBanner.style.display = 'none';
        }
      }
    } catch (err) {
      console.warn('Could not fetch runtime config:', err.message || err);
      configStatus.textContent = 'Config fetch failed';
    }
  }

  async function applyRuntimeConfig() {
    const payload = {
      preprocess_mode: settingsPreprocess.value,
      force_classification: settingsForce.checked,
      min_confidence: Number(settingsMinConfidence.value) || 0.45,
      min_margin: Number(settingsMinMargin.value) || 0.08,
      last_conv_layer_override: settingsLastConv.value || null,
    };
    try {
      const res = await fetch('http://127.0.0.1:5000/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to apply');
      configStatus.textContent = 'Config applied';
      setTimeout(() => (configStatus.textContent = ''), 2500);
    } catch (err) {
      console.warn('Failed to apply config', err.message || err);
      configStatus.textContent = 'Apply failed';
    }
  }

  async function reloadModel() {
    try {
      btnReloadModel.disabled = true;
      configStatus.textContent = 'Reloading...';
      const res = await fetch('http://127.0.0.1:5000/reload_model', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Reload failed');
      configStatus.textContent = data.mock_inference ? 'MOCK mode' : 'Model reloaded';
      setTimeout(() => (configStatus.textContent = ''), 3000);
    } catch (err) {
      console.warn('Reload model failed', err.message || err);
      configStatus.textContent = 'Reload failed';
    } finally {
      btnReloadModel.disabled = false;
    }
  }

  if (btnApplyConfig) btnApplyConfig.addEventListener('click', applyRuntimeConfig);
  if (btnReloadModel) btnReloadModel.addEventListener('click', reloadModel);
  if (btnQuickTest) btnQuickTest.addEventListener('click', async () => {
    try {
      setLoadingState();
      const resp = await fetch('dragonfruit.jpg');
      const blob = await resp.blob();
      const file = new File([blob], 'dragonfruit.jpg', { type: blob.type });
      await processDashboardImage(file);
    } catch (err) {
      console.error('Quick test failed', err);
      alert('Quick test failed: ' + (err.message || err));
    }
  });
  // populate initial values
  fetchRuntimeConfig();

  function updateChartTheme(statusKey) {
    const chartTheme = {
      healthy: { stroke: '#4caf50', shadow: 'drop-shadow(0 0 6px rgba(76, 175, 80, 0.45))' },
      rotten: { stroke: '#f44336', shadow: 'drop-shadow(0 0 6px rgba(244, 67, 54, 0.45))' },
      unknown: { stroke: '#ff9800', shadow: 'drop-shadow(0 0 6px rgba(255, 152, 0, 0.45))' },
    };
    const theme = chartTheme[statusKey] || chartTheme.unknown;
    document.querySelectorAll('.chart-path-anim').forEach((path) => {
      path.style.stroke = theme.stroke;
      path.style.filter = theme.shadow;
      path.style.opacity = '1';
      path.style.animation = 'none';
      path.offsetHeight;
      path.style.animation = null;
    });
  }

  async function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target.result);
      reader.onerror = () => reject(new Error('Failed to read the selected image.'));
      reader.readAsDataURL(file);
    });
  }

  async function loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('Failed to decode the selected image.'));
      image.src = url;
    });
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function colorStopPalette(value, palette) {
    const clamped = clamp01(value);
    const scaled = clamped * (palette.length - 1);
    const index = Math.floor(scaled);
    const nextIndex = Math.min(palette.length - 1, index + 1);
    const mix = scaled - index;
    const start = palette[index];
    const end = palette[nextIndex];

    return [
      Math.round(start[0] + (end[0] - start[0]) * mix),
      Math.round(start[1] + (end[1] - start[1]) * mix),
      Math.round(start[2] + (end[2] - start[2]) * mix),
    ];
  }

  // Fallback function removed - GradCAM should be fully implemented in Python backend
  // If backend fails, show error message instead of generating client-side heatmaps

  async function processDashboardImage(file) {
    if (!file.type.startsWith('image/')) {
      alert('Invalid format! Please upload an image file.');
      return;
    }

    lastUploadedFile = file;
    const previewUrl = await readFileAsDataUrl(file);
    setImageSources(previewUrl);
    setOverlayImage(document.querySelector('.heatmap-overlay-box .heatmap-color-mask'), transparentPixel, 'normal');
    setOverlayImage(document.querySelector('.gradcam-overlay-box .gradcam-color-mask'), transparentPixel, 'normal');
    setLoadingState();

    const loadingOverlay = document.getElementById('analyze-loading-overlay');
    try {
      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
        loadingOverlay.setAttribute('aria-hidden', 'false');
        loadingOverlay.style.pointerEvents = 'auto';
      }
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetchWithRetry(API_URL, { method: 'POST', body: formData }, 2);
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Backend inference failed.');
      }

      const baseImage = data.display_image || previewUrl;
      setImageSources(baseImage);
      // show returned overlays with proper blend mode for clearer hotspots
      setOverlayImage(document.querySelector('.heatmap-overlay-box .heatmap-color-mask'), data.heatmap_overlay || transparentPixel, 'normal');
      setOverlayImage(document.querySelector('.gradcam-overlay-box .gradcam-color-mask'), data.gradcam_overlay || transparentPixel, 'normal');
      // set pointer panel and install a draggable pointer marker
      const pointerBox = document.getElementById('pointer-box');
      if (pointerBox) {
        lastPointerPoint = (Array.isArray(data.focus_points) && data.focus_points.length > 0)
          ? { x: Number(data.focus_points[0].x) || 50, y: Number(data.focus_points[0].y) || 50 }
          : { x: 50, y: 50 };
        renderPointerPanel(pointerBox, baseImage, data.focus_points || []);
      }
      // create or refresh the heatmap legend/explanation
      ensureHeatmapLegend();
      applyFocusedResponse(data);
      // Immediately request a focused crop-based prediction at the top focus point
      try {
        const fp = (Array.isArray(data.focus_points) && data.focus_points.length>0) ? data.focus_points[0] : null;
        if (fp) requestFocusedPrediction({ x: fp.x, y: fp.y });
      } catch (e) { /* ignore */ }
      if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
        loadingOverlay.setAttribute('aria-hidden', 'true');
        loadingOverlay.style.pointerEvents = 'none';
      }
    } catch (error) {
      const loadingOverlay = document.getElementById('analyze-loading-overlay');
      if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
        loadingOverlay.setAttribute('aria-hidden', 'true');
        loadingOverlay.style.pointerEvents = 'none';
      }
      // Show error message instead of fallback - GradCAM should be handled by Python backend
      alert('Backend analysis failed: ' + (error.message || 'Unknown error'));
      console.error('Analysis error:', error);
    }
  }

  dragDropArea.addEventListener('click', () => dashboardFileUpload.click());

  dashboardFileUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      processDashboardImage(e.target.files[0]);
    }
  });

  dragDropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dragDropArea.style.backgroundColor = '#4b5563';
  });

  dragDropArea.addEventListener('dragleave', () => {
    dragDropArea.style.backgroundColor = '#374151';
  });

  dragDropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDropArea.style.backgroundColor = '#374151';
    if (e.dataTransfer.files.length > 0) {
      processDashboardImage(e.dataTransfer.files[0]);
    }
  });

  // Removed individual badge click handlers — UI now displays only detected status

  // --- SCREEN 3: RESEARCH PAGE SCIENTIFIC PDF CONTROLLER ---
  const pdfScroller = document.getElementById('pdf-scroller');
  const pdfPages = document.querySelectorAll('.pdf-page-sheet');
  
  const pdfNavFirst = document.getElementById('pdf-nav-first');
  const pdfNavPrev = document.getElementById('pdf-nav-prev');
  const pdfNavCurrent = document.getElementById('pdf-nav-current');
  const pdfNavNext = document.getElementById('pdf-nav-next');
  const pdfNavLast = document.getElementById('pdf-nav-last');
  
  const pdfZoomOut = document.getElementById('pdf-zoom-out');
  const pdfZoomLevel = document.getElementById('pdf-zoom-level');
  const pdfZoomIn = document.getElementById('pdf-zoom-in');

  const pdfUtilityIcons = document.querySelector('.pdf-utility-icons');

  let currentZoom = 1.0;
  const zoomStep = 0.1;
  const minZoom = 0.7;
  const maxZoom = 1.5;
  const totalPages = pdfPages.length; // 6 pages

  // 1. Page Indicator Scroll Listener
  if (pdfScroller && pdfPages.length > 0) {
    pdfScroller.addEventListener('scroll', () => {
      const scrollerRect = pdfScroller.getBoundingClientRect();
      const scrollerMid = scrollerRect.top + scrollerRect.height / 2;
      
      let activePage = 1;
      let minDistance = Infinity;
      
      pdfPages.forEach((page, index) => {
        const pageRect = page.getBoundingClientRect();
        const pageMid = pageRect.top + pageRect.height / 2;
        const distance = Math.abs(pageMid - scrollerMid);
        
        if (distance < minDistance) {
          minDistance = distance;
          activePage = index + 1;
        }
      });
      
      if (pdfNavCurrent) {
        pdfNavCurrent.textContent = activePage;
      }
    });
  }

  // Helper function to smooth scroll to a specific page index (0-indexed)
  function scrollToPage(pageIndex) {
    if (pageIndex >= 0 && pageIndex < totalPages && pdfPages[pageIndex]) {
      pdfPages[pageIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Get active page index (0-indexed) based on text in indicator
  function getActivePageIndex() {
    if (pdfNavCurrent) {
      const pageVal = parseInt(pdfNavCurrent.textContent, 10);
      if (!isNaN(pageVal)) {
        return pageVal - 1;
      }
    }
    return 0;
  }

  // Bind navigation arrows
  if (pdfNavFirst) {
    pdfNavFirst.addEventListener('click', () => scrollToPage(0));
  }
  if (pdfNavLast) {
    pdfNavLast.addEventListener('click', () => scrollToPage(totalPages - 1));
  }
  if (pdfNavPrev) {
    pdfNavPrev.addEventListener('click', () => {
      const activeIdx = getActivePageIndex();
      scrollToPage(Math.max(0, activeIdx - 1));
    });
  }
  if (pdfNavNext) {
    pdfNavNext.addEventListener('click', () => {
      const activeIdx = getActivePageIndex();
      scrollToPage(Math.min(totalPages - 1, activeIdx + 1));
    });
  }

  // 2. Zoom Controls
  function applyZoom(zoomVal) {
    currentZoom = Math.max(minZoom, Math.min(maxZoom, zoomVal));
    if (pdfScroller) {
      pdfScroller.style.setProperty('--pdf-zoom', currentZoom);
    }
    if (pdfZoomLevel) {
      pdfZoomLevel.textContent = `${Math.round(currentZoom * 100)}%`;
    }
  }

  if (pdfZoomIn) {
    pdfZoomIn.addEventListener('click', () => {
      applyZoom(currentZoom + zoomStep);
    });
  }
  if (pdfZoomOut) {
    pdfZoomOut.addEventListener('click', () => {
      applyZoom(currentZoom - zoomStep);
    });
  }

  // 3. Utility Icons Click Handlers
  if (pdfUtilityIcons) {
    const svgs = pdfUtilityIcons.querySelectorAll('svg');
    
    // First icon: Download
    if (svgs[0]) {
      svgs[0].addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = './CeileGuce_ComparisonPaper.pdf';
        link.download = 'CeileGuce_ComparisonPaper.pdf';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
      svgs[0].style.cursor = 'pointer';
      svgs[0].setAttribute('title', 'Download PDF');
    }
    
    // Second icon: Print
    if (svgs[1]) {
      svgs[1].addEventListener('click', () => {
        window.open('./CeileGuce_ComparisonPaper.pdf', '_blank');
      });
      svgs[1].style.cursor = 'pointer';
      svgs[1].setAttribute('title', 'Print PDF / Open in New Tab');
    }
  }

});
