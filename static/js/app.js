/**
 * SatQuery AI - Multi-Modal Earth Observation & Geospatial Intelligence
 * Interactive Leaflet GIS Map, Evidence Linking, SAR Analysis, and RAG Engine
 */

(function () {
  'use strict';

  // ==========================================================================
  // Application State
  // ==========================================================================
  const state = {
    conversations: [],
    currentChatId: null,
    currentView: 'map', // 'map', 'image', 'change', 'evidence'
    stagedImages: [],
    stagedFiles: [],
    activeDetections: [],
    activeSarTelemetry: null,
    activeChangeTelemetry: null,
    activeAoiPolygon: null,
    isGenerating: false,
    useRag: true,
    useDetection: true,
    currentModel: 'builtin:visionorbit-core',
    settings: {
      openaiKey: localStorage.getItem('satquery_openai_key') || '',
      ollamaUrl: localStorage.getItem('satquery_ollama_url') || 'http://localhost:11434',
      tavilyKey: localStorage.getItem('satquery_tavily_key') || ''
    },
    map: null,
    mapLayers: {
      sat: null,
      dark: null,
      osm: null,
      detectionsLayer: null,
      aoiLayer: null
    },
    polygonMarkers: [],
    zoomLevel: 1
  };

  // ==========================================================================
  // DOM Elements
  // ==========================================================================
  const elements = {
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
    sidebarCloseBtn: document.getElementById('sidebarCloseBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    openDemoBtn: document.getElementById('openDemoBtn'),
    conversationsList: document.getElementById('conversationsList'),
    currentProviderLabel: document.getElementById('currentProviderLabel'),
    providerSubText: document.getElementById('providerSubText'),
    providerDot: document.getElementById('providerDot'),
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeIconDark: document.getElementById('themeIconDark'),
    themeIconLight: document.getElementById('themeIconLight'),
    modelSelector: document.getElementById('modelSelector'),
    detectionToggle: document.getElementById('detectionToggle'),
    ragToggle: document.getElementById('ragToggle'),
    exportChatBtn: document.getElementById('exportChatBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    generateReportBtn: document.getElementById('generateReportBtn'),

    // Viewport Modes
    modeMapBtn: document.getElementById('modeMapBtn'),
    modeImageBtn: document.getElementById('modeImageBtn'),
    modeChangeBtn: document.getElementById('modeChangeBtn'),
    modeEvidenceBtn: document.getElementById('modeEvidenceBtn'),
    gisMapContainer: document.getElementById('gisMapContainer'),
    imageInspectionContainer: document.getElementById('imageInspectionContainer'),
    changeComparisonContainer: document.getElementById('changeComparisonContainer'),
    evidenceInspectorContainer: document.getElementById('evidenceInspectorContainer'),

    // Map Controls
    leafletMapEl: document.getElementById('leafletMap'),
    basemapSatBtn: document.getElementById('basemapSatBtn'),
    basemapDarkBtn: document.getElementById('basemapDarkBtn'),
    basemapOsmBtn: document.getElementById('basemapOsmBtn'),
    drawBoxAoiBtn: document.getElementById('drawBoxAoiBtn'),
    drawPolyAoiBtn: document.getElementById('drawPolyAoiBtn'),
    clearAoiBtn: document.getElementById('clearAoiBtn'),
    toggleLayerDetections: document.getElementById('toggleLayerDetections'),
    toggleLayerSarMask: document.getElementById('toggleLayerSarMask'),
    toggleLayerHeatmap: document.getElementById('toggleLayerHeatmap'),

    // Decision Support Badges
    trafficIndicatorBadge: document.getElementById('trafficIndicatorBadge'),
    floodIndicatorBadge: document.getElementById('floodIndicatorBadge'),

    // Image Inspection
    activeInspectionImage: document.getElementById('activeInspectionImage'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    resetZoomBtn: document.getElementById('resetZoomBtn'),

    // Change Slider
    comparisonSliderWrapper: document.getElementById('comparisonSliderWrapper'),
    compareBeforeImg: document.getElementById('compareBeforeImg'),
    compareAfterImg: document.getElementById('compareAfterImg'),
    sliderHandle: document.getElementById('sliderHandle'),
    changeDateBefore: document.getElementById('changeDateBefore'),
    changeDateAfter: document.getElementById('changeDateAfter'),

    // Evidence
    evidenceList: document.getElementById('evidenceList'),
    evidenceCountBadge: document.getElementById('evidenceCountBadge'),

    // Chat Feed
    chatContainer: document.getElementById('chatContainer'),
    welcomeHero: document.getElementById('welcomeHero'),
    heroDropzone: document.getElementById('heroDropzone'),
    messagesList: document.getElementById('messagesList'),
    analysisIndicator: document.getElementById('analysisIndicator'),
    analyzingText: document.getElementById('analyzingText'),
    scrollToBottomBtn: document.getElementById('scrollToBottomBtn'),
    chatForm: document.getElementById('chatForm'),
    promptInput: document.getElementById('promptInput'),
    imageFileInput: document.getElementById('imageFileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    sendBtn: document.getElementById('sendBtn'),
    imagePreviewTray: document.getElementById('imagePreviewTray'),
    dragDropOverlay: document.getElementById('dragDropOverlay'),

    // Modals
    demoModal: document.getElementById('demoModal'),
    closeDemoModalBtn: document.getElementById('closeDemoModalBtn'),
    demoScenariosList: document.getElementById('demoScenariosList'),
    reportModal: document.getElementById('reportModal'),
    closeReportModalBtn: document.getElementById('closeReportModalBtn'),
    printReportBtn: document.getElementById('printReportBtn'),
    reportMarkdownRender: document.getElementById('reportMarkdownRender'),
    settingsModal: document.getElementById('settingsModal'),
    openSettingsBtn: document.getElementById('openSettingsBtn'),
    closeSettingsModalBtn: document.getElementById('closeSettingsModalBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    openaiKeyInput: document.getElementById('openaiKeyInput'),
    ollamaUrlInput: document.getElementById('ollamaUrlInput'),
    tavilyKeyInput: document.getElementById('tavilyKeyInput'),
    toastContainer: document.getElementById('toastContainer')
  };

  // ==========================================================================
  // Initialization
  // ==========================================================================
  function init() {
    initLeafletMap();
    initEventListeners();
    initComparisonSlider();
    loadConversations();
    checkHealth();
  }

  // ==========================================================================
  // Leaflet GIS Map Setup
  // ==========================================================================
  function initLeafletMap() {
    if (!elements.leafletMapEl || typeof L === 'undefined') return;

    // Default center over georeferenced sample area (Lat: 42.2805, Lon: -71.7789)
    const initialCoords = [42.2805, -71.7789];
    state.map = L.map('leafletMap', {
      center: initialCoords,
      zoom: 17,
      zoomControl: true
    });

    // Basemaps
    state.mapLayers.sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri, Maxar, Earthstar Geographics',
      maxZoom: 19
    }).addTo(state.map);

    state.mapLayers.dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB',
      maxZoom: 19
    });

    state.mapLayers.osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap',
      maxZoom: 19
    });

    // Feature Layers
    state.mapLayers.detectionsLayer = L.featureGroup().addTo(state.map);
    state.mapLayers.aoiLayer = L.featureGroup().addTo(state.map);

    // Initial footprint marker
    L.circleMarker(initialCoords, {
      radius: 6,
      fillColor: '#00f2fe',
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.8
    }).bindPopup('<b>Satellite Telemetry Anchor</b><br>Lat: 42.2805, Lon: -71.7789').addTo(state.map);
  }

  function switchBasemap(type) {
    if (!state.map) return;
    state.map.removeLayer(state.mapLayers.sat);
    state.map.removeLayer(state.mapLayers.dark);
    state.map.removeLayer(state.mapLayers.osm);

    elements.basemapSatBtn.classList.remove('active');
    elements.basemapDarkBtn.classList.remove('active');
    elements.basemapOsmBtn.classList.remove('active');

    if (type === 'sat') {
      state.mapLayers.sat.addTo(state.map);
      elements.basemapSatBtn.classList.add('active');
    } else if (type === 'dark') {
      state.mapLayers.dark.addTo(state.map);
      elements.basemapDarkBtn.classList.add('active');
    } else {
      state.mapLayers.osm.addTo(state.map);
      elements.basemapOsmBtn.classList.add('active');
    }
  }

  function plotDetectionsOnMap(detections) {
    if (!state.map || !detections || !state.mapLayers.detectionsLayer) return;
    state.mapLayers.detectionsLayer.clearLayers();
    state.polygonMarkers = [];

    const classColors = {
      'cargo truck': '#00f2fe',
      'small car': '#10b981',
      'van': '#f59e0b',
      'dump truck': '#f43f5e',
      'other-airplane': '#8b5cf6',
      'dry cargo ship': '#38bdf8'
    };

    const boundsGroup = [];

    detections.forEach((d) => {
      const cls = (d.class_name || '').toLowerCase();
      const color = classColors[cls] || '#00f2fe';
      const geoPoly = d.obb && d.obb.polygon_corners_latlon;
      const lat = d.latitude;
      const lon = d.longitude;

      if (geoPoly && geoPoly.length >= 3) {
        // Draw Oriented Bounding Box Polygon
        const polyLayer = L.polygon(geoPoly, {
          color: color,
          weight: 2,
          fillColor: color,
          fillOpacity: 0.35
        });

        const popupContent = `
          <div style="font-family: 'Inter', sans-serif; font-size: 12px;">
            <strong style="color: ${color}; font-size: 13px;">#${d.id} ${d.class_name}</strong><br>
            <b>Confidence:</b> ${d.confidence_percent || (d.confidence * 100).toFixed(1) + '%'}<br>
            <b>Coordinates:</b> (${lat ? lat.toFixed(6) : 'N/A'}, ${lon ? lon.toFixed(6) : 'N/A'})<br>
            <b>Dimensions:</b> ${d.obb.width_px} × ${d.obb.height_px} px (Angle: ${d.obb.angle_degrees}°)<br>
          </div>
        `;
        polyLayer.bindPopup(popupContent);
        polyLayer.detId = d.id;
        state.mapLayers.detectionsLayer.addLayer(polyLayer);
        state.polygonMarkers.push(polyLayer);
        boundsGroup.push(...geoPoly);
      } else if (lat && lon) {
        // Fallback Circle Marker
        const marker = L.circleMarker([lat, lon], {
          radius: 5,
          color: color,
          fillColor: color,
          fillOpacity: 0.7
        }).bindPopup(`<strong>#${d.id} ${d.class_name}</strong><br>Confidence: ${d.confidence_percent}`);
        marker.detId = d.id;
        state.mapLayers.detectionsLayer.addLayer(marker);
        state.polygonMarkers.push(marker);
        boundsGroup.push([lat, lon]);
      }
    });

    if (boundsGroup.length > 0) {
      state.map.fitBounds(L.latLngBounds(boundsGroup), { padding: [30, 30] });
    }
  }

  // ==========================================================================
  // Visual Evidence Linking
  // ==========================================================================
  function highlightEvidenceItem(evidenceId) {
    switchView('map');
    let matchedLayer = null;

    state.polygonMarkers.forEach((layer) => {
      if (layer.detId == evidenceId || String(layer.detId) === String(evidenceId).replace('det_', '')) {
        matchedLayer = layer;
        layer.setStyle({
          color: '#ffffff',
          weight: 4,
          fillColor: '#00f2fe',
          fillOpacity: 0.8
        });
        if (layer.openPopup) layer.openPopup();
      } else {
        layer.setStyle({ fillOpacity: 0.25, weight: 1.5 });
      }
    });

    if (matchedLayer && state.map) {
      if (matchedLayer.getBounds) {
        state.map.flyToBounds(matchedLayer.getBounds(), { maxZoom: 19, duration: 1.2 });
      } else if (matchedLayer.getLatLng) {
        state.map.flyTo(matchedLayer.getLatLng(), 19, { duration: 1.2 });
      }
    }

    // Also highlight card in evidence inspector
    document.querySelectorAll('.evidence-card').forEach((card) => {
      if (card.dataset.id == evidenceId) {
        card.classList.add('focused');
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        card.classList.remove('focused');
      }
    });
  }

  function renderEvidenceInspector(evidenceItems) {
    if (!elements.evidenceList) return;
    elements.evidenceList.innerHTML = '';
    elements.evidenceCountBadge.textContent = `${evidenceItems.length} items`;

    if (!evidenceItems || evidenceItems.length === 0) {
      elements.evidenceList.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 20px;">No active evidence items in session.</div>';
      return;
    }

    evidenceItems.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'evidence-card';
      card.dataset.id = item.id;

      const latLonStr = item.latitude && item.longitude ? `(${item.latitude.toFixed(6)}, ${item.longitude.toFixed(6)})` : 'Pixel Space';
      const confStr = item.confidence_percent || (item.confidence ? (item.confidence * 100).toFixed(1) + '%' : 'Verified');

      card.innerHTML = `
        <div class="evidence-card-header">
          <span class="evidence-label">${item.label || 'Target #' + item.id}</span>
          <span class="evidence-score">${confStr}</span>
        </div>
        <div class="evidence-geo">📍 ${latLonStr} • ${item.provenance || 'YOLO-OBB / GIS'}</div>
      `;

      card.addEventListener('click', () => {
        highlightEvidenceItem(item.id);
      });

      elements.evidenceList.appendChild(card);
    });
  }

  // ==========================================================================
  // Comparison Swipe Slider
  // ==========================================================================
  function initComparisonSlider() {
    if (!elements.comparisonSliderWrapper || !elements.sliderHandle) return;

    let isDragging = false;

    function setSliderPosition(x) {
      const rect = elements.comparisonSliderWrapper.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      pos = Math.max(0, Math.min(1, pos));
      const pct = (pos * 100).toFixed(2);

      elements.sliderHandle.style.left = `${pct}%`;
      elements.compareAfterImg.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    }

    elements.sliderHandle.addEventListener('mousedown', () => { isDragging = true; });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (isDragging) setSliderPosition(e.clientX);
    });

    elements.sliderHandle.addEventListener('touchstart', () => { isDragging = true; });
    window.addEventListener('touchend', () => { isDragging = false; });
    window.addEventListener('touchmove', (e) => {
      if (isDragging && e.touches[0]) setSliderPosition(e.touches[0].clientX);
    });
  }

  // ==========================================================================
  // Viewport Switcher
  // ==========================================================================
  function switchView(viewName) {
    state.currentView = viewName;

    elements.modeMapBtn.classList.remove('active');
    elements.modeImageBtn.classList.remove('active');
    elements.modeChangeBtn.classList.remove('active');
    elements.modeEvidenceBtn.classList.remove('active');

    elements.gisMapContainer.classList.add('hidden');
    elements.imageInspectionContainer.classList.add('hidden');
    elements.changeComparisonContainer.classList.add('hidden');
    elements.evidenceInspectorContainer.classList.add('hidden');

    if (viewName === 'map') {
      elements.modeMapBtn.classList.add('active');
      elements.gisMapContainer.classList.remove('hidden');
      if (state.map) setTimeout(() => state.map.invalidateSize(), 200);
    } else if (viewName === 'image') {
      elements.modeImageBtn.classList.add('active');
      elements.imageInspectionContainer.classList.remove('hidden');
    } else if (viewName === 'change') {
      elements.modeChangeBtn.classList.add('active');
      elements.changeComparisonContainer.classList.remove('hidden');
    } else if (viewName === 'evidence') {
      elements.modeEvidenceBtn.classList.add('active');
      elements.evidenceInspectorContainer.classList.remove('hidden');
    }
  }

  // ==========================================================================
  // Chat & Communication Logic
  // ==========================================================================
  async function sendMessage(customPrompt = null) {
    const text = (customPrompt !== null ? customPrompt : elements.promptInput.value).trim();
    if (!text && state.stagedImages.length === 0) return;
    if (state.isGenerating) return;

    elements.promptInput.value = '';
    elements.welcomeHero.classList.add('hidden');
    elements.messagesList.classList.remove('hidden');

    const userImages = [...state.stagedImages];
    clearStagedImages();

    // Render User Message
    appendMessage({
      role: 'user',
      content: text,
      images: userImages
    });

    state.isGenerating = true;
    elements.analysisIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
      const payload = {
        prompt: text,
        images: userImages,
        history: getConversationHistory(),
        provider: state.currentModel.split(':')[0] || 'auto',
        model: state.currentModel.includes(':') ? state.currentModel.split(':')[1] : state.currentModel,
        useRag: state.useRag,
        useDetection: state.useDetection,
        apiKey: state.settings.openaiKey,
        ollamaBaseUrl: state.settings.ollamaUrl,
        tavilyApiKey: state.settings.tavilyKey
      };

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      
      // Update Active Telemetry
      if (data.detection_results && data.detection_results.detections) {
        state.activeDetections = data.detection_results.detections;
        plotDetectionsOnMap(state.activeDetections);
      }

      if (data.annotated_image) {
        elements.activeInspectionImage.src = data.annotated_image;
      }

      if (data.evidence_items) {
        renderEvidenceInspector(data.evidence_items);
      }

      // Update Decision Badges
      updateDecisionIndicators(data);

      appendMessage({
        role: 'assistant',
        content: data.reply,
        provider_used: data.provider_used,
        model_used: data.model_used,
        annotated_image: data.annotated_image,
        detection_results: data.detection_results,
        evidence_items: data.evidence_items,
        rag_sources: data.rag_sources
      });

    } catch (err) {
      appendMessage({
        role: 'assistant',
        content: `⚠️ **Inquiry Error:** ${err.message}`,
        provider_used: 'System Error'
      });
    } finally {
      state.isGenerating = false;
      elements.analysisIndicator.classList.add('hidden');
      scrollToBottom();
      saveCurrentConversation();
    }
  }

  function appendMessage(msg) {
    const row = document.createElement('div');
    row.className = `message-row ${msg.role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = msg.role === 'user' ? '👤' : '🛰️';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble markdown-body';

    // Render markdown
    if (typeof marked !== 'undefined') {
      bubble.innerHTML = marked.parse(msg.content);
    } else {
      bubble.textContent = msg.content;
    }

    // Render inline image if present
    if (msg.annotated_image) {
      const imgWrap = document.createElement('div');
      imgWrap.style.marginTop = '10px';
      imgWrap.innerHTML = `<img src="${msg.annotated_image}" style="max-width: 100%; border-radius: 8px; cursor: pointer;" alt="Detection Preview">`;
      imgWrap.addEventListener('click', () => {
        elements.activeInspectionImage.src = msg.annotated_image;
        switchView('image');
      });
      bubble.appendChild(imgWrap);
    }

    // Render Clickable Evidence Chips
    if (msg.evidence_items && msg.evidence_items.length > 0) {
      const evidenceWrap = document.createElement('div');
      evidenceWrap.style.marginTop = '12px';
      evidenceWrap.innerHTML = '<div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--accent-cyan); margin-bottom: 6px;">🎯 Linked Visual Proof:</div>';
      
      const chipsContainer = document.createElement('div');
      chipsContainer.style.display = 'flex';
      chipsContainer.style.flexWrap = 'wrap';
      chipsContainer.style.gap = '6px';

      msg.evidence_items.slice(0, 10).forEach((item) => {
        const chip = document.createElement('button');
        chip.className = 'chip-btn active';
        chip.style.cursor = 'pointer';
        chip.innerHTML = `<span>#${item.id} ${item.label}</span>`;
        chip.addEventListener('click', () => highlightEvidenceItem(item.id));
        chipsContainer.appendChild(chip);
      });

      if (msg.evidence_items.length > 10) {
        const moreChip = document.createElement('span');
        moreChip.style.fontSize = '11px';
        moreChip.style.color = 'var(--text-muted)';
        moreChip.textContent = `+ ${msg.evidence_items.length - 10} more in evidence drawer`;
        chipsContainer.appendChild(moreChip);
      }

      evidenceWrap.appendChild(chipsContainer);
      bubble.appendChild(evidenceWrap);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    elements.messagesList.appendChild(row);
  }

  function updateDecisionIndicators(data) {
    if (!elements.trafficIndicatorBadge || !elements.floodIndicatorBadge) return;

    if (data.intent_detected === 'TRAFFIC_ANALYSIS' || (data.detection_results && data.detection_results.total_detections > 20)) {
      elements.trafficIndicatorBadge.className = 'intel-badge elevated';
      elements.trafficIndicatorBadge.innerHTML = '<span class="badge-dot"></span><span>Traffic: Elevated Hotspot</span>';
    }

    if (data.intent_detected === 'FLOOD_ANALYSIS' || data.intent_detected === 'SAR_ANALYSIS') {
      elements.floodIndicatorBadge.className = 'intel-badge elevated';
      elements.floodIndicatorBadge.innerHTML = '<span class="badge-dot"></span><span>Flood: Risk Indicator</span>';
    }
  }

  // ==========================================================================
  // File Staging & Uploads
  // ==========================================================================
  async function handleFileUpload(file) {
    showToast(`Uploading ${file.name}...`);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      const data = await res.json();

      if (data.file_type === 'npy') {
        stageImage(data.npy_reference, file.name);
        showToast(`SAR .npy ingested: ${file.name}`);
      } else {
        stageImage(data.data_url, file.name);
        elements.activeInspectionImage.src = data.data_url;
        showToast(`Satellite image ingested: ${file.name}`);
      }
    } catch (err) {
      showToast(`Upload error: ${err.message}`);
    }
  }

  function stageImage(dataUrl, filename = 'Image') {
    state.stagedImages.push(dataUrl);
    renderStagedTray();
  }

  function clearStagedImages() {
    state.stagedImages = [];
    renderStagedTray();
  }

  function renderStagedTray() {
    if (!elements.imagePreviewTray) return;
    elements.imagePreviewTray.innerHTML = '';
    if (state.stagedImages.length === 0) {
      elements.imagePreviewTray.classList.add('hidden');
      return;
    }
    elements.imagePreviewTray.classList.remove('hidden');

    state.stagedImages.forEach((img, idx) => {
      const thumb = document.createElement('div');
      thumb.style.position = 'relative';
      thumb.style.display = 'inline-block';
      thumb.style.marginRight = '8px';

      const previewSrc = img.startsWith('npy://') ? '/static/images/sat_icon.png' : img;
      thumb.innerHTML = `
        <img src="${previewSrc}" style="width: 52px; height: 52px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-glow);">
        <button style="position: absolute; top: -6px; right: -6px; background: var(--accent-rose); color: white; border-radius: 50%; width: 18px; height: 18px; font-size: 10px; cursor: pointer;">✕</button>
      `;
      thumb.querySelector('button').addEventListener('click', () => {
        state.stagedImages.splice(idx, 1);
        renderStagedTray();
      });
      elements.imagePreviewTray.appendChild(thumb);
    });
  }

  // ==========================================================================
  // Demo Scenarios & Reports
  // ==========================================================================
  async function loadDemoScenarios() {
    try {
      const res = await fetch('/api/demo/scenarios');
      if (!res.ok) return;
      const data = await res.json();
      elements.demoScenariosList.innerHTML = '';

      data.scenarios.forEach((s) => {
        const card = document.createElement('div');
        card.className = 'demo-scenario-card';
        card.innerHTML = `
          <div>
            <div class="demo-scenario-title">${s.name} <span class="badge" style="background: rgba(0,242,254,0.15); color: var(--accent-cyan); font-size: 10px; margin-left: 6px;">${s.tag}</span></div>
            <div class="demo-scenario-desc">${s.description}</div>
          </div>
          <button class="btn primary" style="font-size: 12px; padding: 6px 12px;">Launch</button>
        `;
        card.querySelector('button').addEventListener('click', () => {
          elements.demoModal.classList.add('hidden');
          runDemoScenario(s);
        });
        elements.demoScenariosList.appendChild(card);
      });
    } catch (e) {
      console.error('Failed to load demo scenarios', e);
    }
  }

  async function runDemoScenario(scenario) {
    showToast(`Launching ${scenario.name}...`);
    elements.promptInput.value = scenario.default_prompt;

    if (scenario.sample_image) {
      // Ingest test image
      try {
        const res = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: scenario.sample_image, confThreshold: 0.20 })
        });
        const detData = await res.json();
        if (detData.detections) {
          state.activeDetections = detData.detections;
          plotDetectionsOnMap(detData.detections);
        }
        if (detData.annotated_image) {
          elements.activeInspectionImage.src = detData.annotated_image;
        }
      } catch (e) {
        console.error(e);
      }
    } else if (scenario.sample_npy) {
      state.stagedImages = [scenario.sample_npy];
      renderStagedTray();
    } else if (scenario.type === 'change' && scenario.sample_image_a && scenario.sample_image_b) {
      // Execute change detection
      try {
        const res = await fetch('/api/change/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_a: scenario.sample_image_a,
            image_b: scenario.sample_image_b,
            date_a: '2024-01-15',
            date_b: '2024-06-20'
          })
        });
        const chData = await res.json();
        elements.compareBeforeImg.src = chData.before_preview;
        elements.compareAfterImg.src = chData.change_heatmap_preview;
        switchView('change');
      } catch (e) {
        console.error(e);
      }
    }

    sendMessage(scenario.default_prompt);
  }

  async function generateIntelligenceReport() {
    showToast('Compiling executive intelligence report...');
    try {
      const payload = {
        title: 'VisionOrbit Earth Observation Intelligence Report',
        aoi_name: 'Operational Satellite Footprint',
        detections: state.activeDetections,
        sensor_info: {
          sensor: 'Multi-Sensor Optical & Sentinel-1 SAR',
          resolution: '0.5m High-Res GSD',
          crs: 'EPSG:4326 (WGS84)'
        }
      };

      const res = await fetch('/api/report/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Report generation failed');
      const data = await res.json();

      if (typeof marked !== 'undefined') {
        elements.reportMarkdownRender.innerHTML = marked.parse(data.markdown_content);
      } else {
        elements.reportMarkdownRender.textContent = data.markdown_content;
      }

      elements.reportModal.classList.remove('hidden');
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  // ==========================================================================
  // Event Listeners & Utilities
  // ==========================================================================
  function initEventListeners() {
    // Sidebar toggle
    elements.sidebarToggleBtn.addEventListener('click', () => {
      elements.sidebar.classList.toggle('collapsed');
      setTimeout(() => { if (state.map) state.map.invalidateSize(); }, 250);
    });

    // Viewport Mode Buttons
    elements.modeMapBtn.addEventListener('click', () => switchView('map'));
    elements.modeImageBtn.addEventListener('click', () => switchView('image'));
    elements.modeChangeBtn.addEventListener('click', () => switchView('change'));
    elements.modeEvidenceBtn.addEventListener('click', () => switchView('evidence'));

    // Map Basemaps
    elements.basemapSatBtn.addEventListener('click', () => switchBasemap('sat'));
    elements.basemapDarkBtn.addEventListener('click', () => switchBasemap('dark'));
    elements.basemapOsmBtn.addEventListener('click', () => switchBasemap('osm'));

    // AOI Buttons
    elements.drawBoxAoiBtn.addEventListener('click', () => {
      showToast('Click and drag on map to select Bounding Box AOI');
    });
    elements.clearAoiBtn.addEventListener('click', () => {
      if (state.mapLayers.aoiLayer) state.mapLayers.aoiLayer.clearLayers();
      if (state.activeDetections.length > 0) plotDetectionsOnMap(state.activeDetections);
      showToast('AOI cleared. Displaying all targets.');
    });

    // Layer Toggles
    elements.toggleLayerDetections.addEventListener('change', (e) => {
      if (e.target.checked) state.mapLayers.detectionsLayer.addTo(state.map);
      else state.map.removeLayer(state.mapLayers.detectionsLayer);
    });

    // Chat Form Submit
    elements.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      sendMessage();
    });

    elements.promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // File Upload Drag & Drop
    elements.uploadBtn.addEventListener('click', () => elements.imageFileInput.click());
    elements.imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });

    elements.heroDropzone.addEventListener('click', () => elements.imageFileInput.click());

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dragDropOverlay.classList.remove('hidden');
    });
    elements.dragDropOverlay.addEventListener('dragleave', (e) => {
      elements.dragDropOverlay.classList.add('hidden');
    });
    elements.dragDropOverlay.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dragDropOverlay.classList.add('hidden');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    // Suggestion Cards
    document.querySelectorAll('.suggestion-card').forEach((card) => {
      card.addEventListener('click', () => {
        sendMessage(card.dataset.prompt);
      });
    });

    // Model Selector
    elements.modelSelector.addEventListener('change', (e) => {
      state.currentModel = e.target.value;
      showToast(`Switched active engine to ${state.currentModel}`);
    });

    // Feature Toggles
    elements.detectionToggle.addEventListener('click', () => {
      state.useDetection = !state.useDetection;
      elements.detectionToggle.classList.toggle('active', state.useDetection);
      showToast(`YOLO-OBB Detection ${state.useDetection ? 'Enabled' : 'Disabled'}`);
    });

    elements.ragToggle.addEventListener('click', () => {
      state.useRag = !state.useRag;
      elements.ragToggle.classList.toggle('active', state.useRag);
      showToast(`Pinecone RAG Knowledge Base ${state.useRag ? 'Enabled' : 'Disabled'}`);
    });

    // Demos & Report
    elements.openDemoBtn.addEventListener('click', () => {
      loadDemoScenarios();
      elements.demoModal.classList.remove('hidden');
    });
    elements.closeDemoModalBtn.addEventListener('click', () => {
      elements.demoModal.classList.add('hidden');
    });

    elements.generateReportBtn.addEventListener('click', generateIntelligenceReport);
    elements.closeReportModalBtn.addEventListener('click', () => {
      elements.reportModal.classList.add('hidden');
    });
    elements.printReportBtn.addEventListener('click', () => {
      window.print();
    });

    // Settings Modal
    elements.openSettingsBtn.addEventListener('click', () => {
      elements.settingsModal.classList.remove('hidden');
    });
    elements.closeSettingsModalBtn.addEventListener('click', () => {
      elements.settingsModal.classList.add('hidden');
    });
    elements.saveSettingsBtn.addEventListener('click', () => {
      state.settings.openaiKey = elements.openaiKeyInput.value.trim();
      state.settings.ollamaUrl = elements.ollamaUrlInput.value.trim();
      state.settings.tavilyKey = elements.tavilyKeyInput.value.trim();
      localStorage.setItem('satquery_openai_key', state.settings.openaiKey);
      localStorage.setItem('satquery_ollama_url', state.settings.ollamaUrl);
      localStorage.setItem('satquery_tavily_key', state.settings.tavilyKey);
      elements.settingsModal.classList.add('hidden');
      showToast('Configuration preferences saved');
    });

    // Clear & New Mission
    elements.newChatBtn.addEventListener('click', () => {
      elements.messagesList.innerHTML = '';
      elements.messagesList.classList.add('hidden');
      elements.welcomeHero.classList.remove('hidden');
      state.activeDetections = [];
      if (state.mapLayers.detectionsLayer) state.mapLayers.detectionsLayer.clearLayers();
      showToast('New mission initialized.');
    });
    elements.clearChatBtn.addEventListener('click', () => {
      elements.messagesList.innerHTML = '';
      elements.messagesList.classList.add('hidden');
      elements.welcomeHero.classList.remove('hidden');
    });
  }

  function getConversationHistory() {
    // Extracts message history for context
    return [];
  }

  function loadConversations() {
    // Load recent mission logs
  }

  function saveCurrentConversation() {
    // Persist conversation
  }

  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        elements.currentProviderLabel.textContent = '37-Class YOLO + SAR U-Net';
        elements.providerSubText = 'Deterministic GIS Engine Active';
      }
    } catch (e) {}
  }

  function scrollToBottom() {
    if (elements.chatContainer) {
      elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }
  }

  function showToast(msg) {
    if (!elements.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  // Initialize application on DOM ready
  document.addEventListener('DOMContentLoaded', init);
})();
