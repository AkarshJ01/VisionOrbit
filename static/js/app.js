/**
 * VisionOrbit - Client Application Logic
 * Modern OpenAI-style Multimodal Prompting & Pinecone RAG Intelligence Engine
 */

(function () {
  'use strict';

  // State Management
  const state = {
    conversations: [],
    currentChatId: null,
    stagedImages: [], // Array of base64 data URLs
    isGenerating: false,
    useWebSearch: false,
    useRag: true, // Pinecone RAG enabled by default
    theme: localStorage.getItem('visionorbit_theme') || 'dark',
    settings: {
      openaiKey: localStorage.getItem('visionorbit_openai_key') || '',
      ollamaUrl: localStorage.getItem('visionorbit_ollama_url') || 'http://localhost:11434',
      tavilyKey: localStorage.getItem('visionorbit_tavily_key') || '',
      systemPrompt: localStorage.getItem('visionorbit_sys_prompt') || ''
    },
    currentModel: 'ollama:gpt-oss:20b',
    lightboxScale: 1
  };

  // DOM Elements
  const elements = {
    appLayout: document.querySelector('.app-layout'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
    sidebarCloseBtn: document.getElementById('sidebarCloseBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    conversationsList: document.getElementById('conversationsList'),
    currentProviderLabel: document.getElementById('currentProviderLabel'),
    providerSubText: document.getElementById('providerSubText'),
    providerDot: document.getElementById('providerDot'),
    themeToggleBtn: document.getElementById('themeToggleBtn'),
    themeIconDark: document.getElementById('themeIconDark'),
    themeIconLight: document.getElementById('themeIconLight'),
    
    modelSelector: document.getElementById('modelSelector'),
    ragToggle: document.getElementById('ragToggle'),
    webSearchToggle: document.getElementById('webSearchToggle'),
    exportChatBtn: document.getElementById('exportChatBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),

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
    voiceInputBtn: document.getElementById('voiceInputBtn'),
    sendBtn: document.getElementById('sendBtn'),
    imagePreviewTray: document.getElementById('imagePreviewTray'),
    dragDropOverlay: document.getElementById('dragDropOverlay'),

    // Settings Modal
    settingsModal: document.getElementById('settingsModal'),
    openSettingsBtn: document.getElementById('openSettingsBtn'),
    closeSettingsModalBtn: document.getElementById('closeSettingsModalBtn'),
    openaiKeyInput: document.getElementById('openaiKeyInput'),
    toggleKeyVisibility: document.getElementById('toggleKeyVisibility'),
    ollamaUrlInput: document.getElementById('ollamaUrlInput'),
    tavilyKeyInput: document.getElementById('tavilyKeyInput'),
    systemPromptInput: document.getElementById('systemPromptInput'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    resetSettingsBtn: document.getElementById('resetSettingsBtn'),

    // Lightbox Modal
    imageLightboxModal: document.getElementById('imageLightboxModal'),
    lightboxImage: document.getElementById('lightboxImage'),
    lightboxCloseBtn: document.getElementById('lightboxCloseBtn'),
    lightboxZoomIn: document.getElementById('lightboxZoomIn'),
    lightboxZoomOut: document.getElementById('lightboxZoomOut'),

    toastContainer: document.getElementById('toastContainer')
  };

  // Configure Marked Markdown Renderer
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true,
      highlight: function (code, lang) {
        if (window.hljs) {
          const language = hljs.getLanguage(lang) ? lang : 'plaintext';
          return hljs.highlight(code, { language }).value;
        }
        return code;
      }
    });
  }

  /* ==========================================================================
     Initialization & Persistence
     ========================================================================== */

  function initApp() {
    loadTheme();
    loadSettings();
    loadConversations();
    setupEventListeners();
    fetchModelCapabilities();

    if (elements.promptInput) {
      elements.promptInput.focus();
    }
  }

  function loadTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    if (state.theme === 'light') {
      elements.themeIconDark.classList.add('hidden');
      elements.themeIconLight.classList.remove('hidden');
    } else {
      elements.themeIconDark.classList.remove('hidden');
      elements.themeIconLight.classList.add('hidden');
    }
  }

  function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('visionorbit_theme', state.theme);
    loadTheme();
    showToast(`Switched to ${state.theme} mode`, 'info');
  }

  function loadSettings() {
    elements.openaiKeyInput.value = state.settings.openaiKey;
    elements.ollamaUrlInput.value = state.settings.ollamaUrl;
    elements.tavilyKeyInput.value = state.settings.tavilyKey;
    elements.systemPromptInput.value = state.settings.systemPrompt;
  }

  function saveSettings() {
    state.settings.openaiKey = elements.openaiKeyInput.value.trim();
    state.settings.ollamaUrl = elements.ollamaUrlInput.value.trim() || 'http://localhost:11434';
    state.settings.tavilyKey = elements.tavilyKeyInput.value.trim();
    state.settings.systemPrompt = elements.systemPromptInput.value.trim();

    localStorage.setItem('visionorbit_openai_key', state.settings.openaiKey);
    localStorage.setItem('visionorbit_ollama_url', state.settings.ollamaUrl);
    localStorage.setItem('visionorbit_tavily_key', state.settings.tavilyKey);
    localStorage.setItem('visionorbit_sys_prompt', state.settings.systemPrompt);

    closeModal(elements.settingsModal);
    showToast('Settings saved successfully', 'success');
    fetchModelCapabilities();
  }

  function resetSettings() {
    elements.openaiKeyInput.value = '';
    elements.ollamaUrlInput.value = 'http://localhost:11434';
    elements.tavilyKeyInput.value = '';
    elements.systemPromptInput.value = '';
    saveSettings();
  }

  async function fetchModelCapabilities() {
    try {
      const url = `/api/models?ollama_url=${encodeURIComponent(state.settings.ollamaUrl)}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        updateModelDropdown(data.providers);
      }
    } catch (e) {
      console.warn('Could not refresh model capabilities:', e);
    }
  }

  function updateModelDropdown(providers) {
    if (!elements.modelSelector) return;
    const currentVal = elements.modelSelector.value;
    elements.modelSelector.innerHTML = '';

    providers.forEach(p => {
      const optgroup = document.createElement('optgroup');
      optgroup.label = `${p.name} (${p.badge})`;
      p.models.forEach(m => {
        const option = document.createElement('option');
        option.value = `${p.id}:${m.id}`;
        option.textContent = m.name;
        optgroup.appendChild(option);
      });
      elements.modelSelector.appendChild(optgroup);
    });

    if (currentVal && elements.modelSelector.querySelector(`option[value="${currentVal}"]`)) {
      elements.modelSelector.value = currentVal;
    } else if (elements.modelSelector.querySelector('option[value="ollama:gpt-oss:20b"]')) {
      elements.modelSelector.value = 'ollama:gpt-oss:20b';
    } else {
      elements.modelSelector.value = 'builtin:visionorbit-core';
    }
    handleModelChange();
  }

  function handleModelChange() {
    const val = elements.modelSelector.value;
    state.currentModel = val;
    const [provider, model] = val.split(':');
    
    if (provider === 'ollama') {
      elements.currentProviderLabel.textContent = `Ollama (${model})`;
      elements.providerSubText.textContent = state.useRag ? 'Pinecone RAG Active' : 'Local LLM Inference';
      elements.providerDot.className = 'provider-dot active';
    } else if (provider === 'openai') {
      elements.currentProviderLabel.textContent = `OpenAI (${model})`;
      elements.providerSubText.textContent = state.settings.openaiKey ? 'Live Multimodal Inference' : 'Using Server/Env Key';
      elements.providerDot.className = 'provider-dot active';
    } else {
      elements.currentProviderLabel.textContent = 'VisionOrbit Smart Core';
      elements.providerSubText.textContent = 'Instant Multimodal Analysis';
      elements.providerDot.className = 'provider-dot active';
    }
  }

  /* ==========================================================================
     Conversation History Management
     ========================================================================== */

  function loadConversations() {
    try {
      const saved = localStorage.getItem('visionorbit_conversations');
      state.conversations = saved ? JSON.parse(saved) : [];
    } catch (e) {
      state.conversations = [];
    }

    if (state.conversations.length > 0) {
      switchConversation(state.conversations[0].id);
    } else {
      startNewChat();
    }
    renderConversationsList();
  }

  function saveConversations() {
    localStorage.setItem('visionorbit_conversations', JSON.stringify(state.conversations));
    renderConversationsList();
  }

  function startNewChat() {
    const newChat = {
      id: 'chat_' + Date.now(),
      title: 'New Inquiry',
      messages: [],
      createdAt: Date.now()
    };
    state.conversations.unshift(newChat);
    state.currentChatId = newChat.id;
    state.stagedImages = [];
    renderStagedImages();
    saveConversations();
    renderCurrentChat();
  }

  function switchConversation(chatId) {
    state.currentChatId = chatId;
    state.stagedImages = [];
    renderStagedImages();
    renderConversationsList();
    renderCurrentChat();
  }

  function deleteConversation(chatId, e) {
    if (e) e.stopPropagation();
    state.conversations = state.conversations.filter(c => c.id !== chatId);
    if (state.currentChatId === chatId) {
      if (state.conversations.length > 0) {
        state.currentChatId = state.conversations[0].id;
      } else {
        startNewChat();
        return;
      }
    }
    saveConversations();
    renderCurrentChat();
  }

  function renderConversationsList() {
    if (!elements.conversationsList) return;
    elements.conversationsList.innerHTML = '';

    state.conversations.forEach(chat => {
      const item = document.createElement('div');
      item.className = `chat-history-item ${chat.id === state.currentChatId ? 'active' : ''}`;
      item.onclick = () => switchConversation(chat.id);

      const titleSpan = document.createElement('span');
      titleSpan.className = 'history-item-title';
      titleSpan.textContent = chat.title || 'New Inquiry';

      const delBtn = document.createElement('button');
      delBtn.className = 'history-item-delete';
      delBtn.title = 'Delete conversation';
      delBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
      delBtn.onclick = (e) => deleteConversation(chat.id, e);

      item.appendChild(titleSpan);
      item.appendChild(delBtn);
      elements.conversationsList.appendChild(item);
    });
  }

  function getCurrentChat() {
    return state.conversations.find(c => c.id === state.currentChatId);
  }

  function renderCurrentChat() {
    const chat = getCurrentChat();
    if (!chat || chat.messages.length === 0) {
      elements.welcomeHero.classList.remove('hidden');
      elements.messagesList.classList.add('hidden');
      elements.messagesList.innerHTML = '';
      return;
    }

    elements.welcomeHero.classList.add('hidden');
    elements.messagesList.classList.remove('hidden');
    elements.messagesList.innerHTML = '';

    chat.messages.forEach(msg => {
      appendMessageToDOM(msg);
    });

    scrollChatToBottom();
  }

  /* ==========================================================================
     Image Upload & Staging
     ========================================================================== */

  function handleFileSelection(files) {
    if (!files || files.length === 0) return;

    Array.from(files).forEach(file => {
      if (!file.type.startsWith('image/')) {
        showToast(`Skipped non-image file: ${file.name}`, 'error');
        return;
      }
      if (file.size > 15 * 1024 * 1024) {
        showToast(`Image too large (max 15MB): ${file.name}`, 'error');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        state.stagedImages.push(e.target.result);
        renderStagedImages();
        showToast(`Attached image: ${file.name}`, 'info');
      };
      reader.readAsDataURL(file);
    });
  }

  function renderStagedImages() {
    if (!elements.imagePreviewTray) return;

    if (state.stagedImages.length === 0) {
      elements.imagePreviewTray.classList.add('hidden');
      elements.imagePreviewTray.innerHTML = '';
      return;
    }

    elements.imagePreviewTray.classList.remove('hidden');
    elements.imagePreviewTray.innerHTML = '';

    state.stagedImages.forEach((imgData, index) => {
      const card = document.createElement('div');
      card.className = 'staged-img-card';

      const img = document.createElement('img');
      img.src = imgData;
      img.alt = `Staged visual ${index + 1}`;
      img.onclick = () => openLightbox(imgData);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'staged-img-remove';
      removeBtn.innerHTML = '&times;';
      removeBtn.title = 'Remove image';
      removeBtn.onclick = (e) => {
        e.stopPropagation();
        state.stagedImages.splice(index, 1);
        renderStagedImages();
      };

      card.appendChild(img);
      card.appendChild(removeBtn);
      elements.imagePreviewTray.appendChild(card);
    });
  }

  /* ==========================================================================
     Chat Submission & Multimodal / Pinecone RAG Generation
     ========================================================================== */

  async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    if (state.isGenerating) return;

    const prompt = elements.promptInput.value.trim();
    const images = [...state.stagedImages];

    if (!prompt && images.length === 0) {
      showToast('Please enter a query or attach an image.', 'error');
      return;
    }

    elements.promptInput.value = '';
    elements.promptInput.style.height = 'auto';
    state.stagedImages = [];
    renderStagedImages();

    const chat = getCurrentChat();
    if (!chat) return;

    const userMsg = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: prompt || 'Analyze and describe the attached image(s).',
      images: images,
      timestamp: Date.now()
    };

    if (chat.messages.length === 0) {
      const autoTitle = prompt ? (prompt.length > 32 ? prompt.substring(0, 32) + '...' : prompt) : 'Visual & Knowledge Inquiry';
      chat.title = autoTitle;
      saveConversations();
    }

    chat.messages.push(userMsg);
    saveConversations();

    elements.welcomeHero.classList.add('hidden');
    elements.messagesList.classList.remove('hidden');
    appendMessageToDOM(userMsg);
    scrollChatToBottom();

    await generateAssistantResponse(prompt, images);
  }

  async function generateAssistantResponse(prompt, images) {
    state.isGenerating = true;
    updateSendButtonState();
    elements.analysisIndicator.classList.remove('hidden');
    elements.analyzingText.textContent = state.useRag && images.length === 0
      ? 'Retrieving from Pinecone knowledge base & thinking...'
      : (images.length > 0 ? 'Analyzing multimodal image features & telemetry...' : 'Generating response...');
    scrollChatToBottom();

    const [provider, model] = state.currentModel.split(':');
    const chat = getCurrentChat();

    const historyPayload = chat.messages.slice(0, -1).map(m => ({
      role: m.role,
      content: m.content
    }));

    try {
      const payload = {
        prompt: prompt || 'Describe and analyze the attached image in detail.',
        images: images,
        history: historyPayload,
        provider: provider,
        model: model,
        apiKey: state.settings.openaiKey || null,
        ollamaBaseUrl: state.settings.ollamaUrl || null,
        tavilyApiKey: state.settings.tavilyKey || null,
        useWebSearch: state.useWebSearch,
        useRag: state.useRag,
        systemPrompt: state.settings.systemPrompt || null
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();

      const assistantMsg = {
        id: 'msg_' + Date.now(),
        role: 'assistant',
        content: data.reply,
        provider_used: data.provider_used,
        model_used: data.model_used,
        image_metadata: data.image_metadata || [],
        search_sources: data.search_sources || [],
        rag_sources: data.rag_sources || [],
        timestamp: Date.now()
      };

      chat.messages.push(assistantMsg);
      saveConversations();

      elements.analysisIndicator.classList.add('hidden');
      await streamMessageToDOM(assistantMsg);

    } catch (err) {
      console.error('Chat error:', err);
      elements.analysisIndicator.classList.add('hidden');

      const errorMsg = {
        id: 'msg_' + Date.now(),
        role: 'assistant',
        content: `⚠️ **An error occurred during generation:**\n\n\`${err.message}\`\n\n*Please verify your Pinecone index or model settings in the ⚙️ Settings dialog.*`,
        provider_used: 'Error Handler',
        timestamp: Date.now()
      };

      chat.messages.push(errorMsg);
      saveConversations();
      appendMessageToDOM(errorMsg);
    } finally {
      state.isGenerating = false;
      updateSendButtonState();
      scrollChatToBottom();
    }
  }

  function updateSendButtonState() {
    if (state.isGenerating) {
      elements.sendBtn.disabled = true;
      elements.sendBtn.innerHTML = `<span class="pulse-spark"></span>`;
    } else {
      elements.sendBtn.disabled = false;
      elements.sendBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>`;
    }
  }

  /* ==========================================================================
     DOM Message Rendering & RAG Sources Display
     ========================================================================== */

  function formatMarkdown(rawText) {
    if (window.marked) {
      return marked.parse(rawText);
    }
    return rawText
      .replace(/\n\n/g, '<p></p>')
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
  }

  function renderRagSources(sources) {
    if (!sources || sources.length === 0) return null;

    const container = document.createElement('div');
    container.className = 'rag-sources-card';

    const header = document.createElement('div');
    header.className = 'rag-sources-header';
    header.innerHTML = `
      <span>📚 Pinecone Knowledge Sources (${sources.length} chunks retrieved)</span>
      <span class="accordion-toggle">▾</span>
    `;

    const list = document.createElement('div');
    list.className = 'rag-sources-list';

    sources.forEach(s => {
      const item = document.createElement('div');
      item.className = 'rag-source-item';
      item.innerHTML = `
        <div class="rag-source-meta">
          <span class="rag-source-filename">📄 ${s.filename}</span>
          <span class="rag-source-page">Page ${s.page}</span>
        </div>
        <div class="rag-source-snippet">${s.snippet}</div>
      `;
      list.appendChild(item);
    });

    header.onclick = () => {
      list.classList.toggle('hidden');
      const toggleIcon = header.querySelector('.accordion-toggle');
      if (toggleIcon) {
        toggleIcon.textContent = list.classList.contains('hidden') ? '▸' : '▾';
      }
    };

    container.appendChild(header);
    container.appendChild(list);
    return container;
  }

  function appendMessageToDOM(msg) {
    const item = document.createElement('div');
    item.className = `message-item ${msg.role}`;
    item.id = msg.id;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    if (msg.role === 'user') {
      avatar.textContent = 'U';
    } else {
      avatar.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"></path></svg>`;
    }

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (msg.images && msg.images.length > 0) {
      const imgStrip = document.createElement('div');
      imgStrip.className = 'msg-images-strip';
      msg.images.forEach((imgData, i) => {
        const thumbWrap = document.createElement('div');
        thumbWrap.className = 'msg-thumb-wrapper';
        thumbWrap.onclick = () => openLightbox(imgData);

        const img = document.createElement('img');
        img.src = imgData;
        img.alt = `Uploaded visual ${i + 1}`;

        const badge = document.createElement('span');
        badge.className = 'thumb-zoom-badge';
        badge.textContent = '🔍 Click to Zoom';

        thumbWrap.appendChild(img);
        thumbWrap.appendChild(badge);
        imgStrip.appendChild(thumbWrap);
      });
      bubble.appendChild(imgStrip);
    }

    const textCard = document.createElement('div');
    textCard.className = 'msg-text-card markdown-body';
    textCard.innerHTML = formatMarkdown(msg.content);
    enhanceCodeBlocks(textCard);
    bubble.appendChild(textCard);

    // Render RAG sources if present
    if (msg.rag_sources && msg.rag_sources.length > 0) {
      const ragCard = renderRagSources(msg.rag_sources);
      if (ragCard) bubble.appendChild(ragCard);
    }

    if (msg.role === 'assistant') {
      const actionsBar = document.createElement('div');
      actionsBar.className = 'msg-actions-bar';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'action-pill-btn';
      copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> <span>Copy</span>`;
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(msg.content);
        showToast('Copied response to clipboard', 'success');
      };

      const speakBtn = document.createElement('button');
      speakBtn.className = 'action-pill-btn';
      speakBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> <span>Listen</span>`;
      speakBtn.onclick = () => speakText(msg.content);

      actionsBar.appendChild(copyBtn);
      actionsBar.appendChild(speakBtn);

      if (msg.provider_used) {
        const modelTag = document.createElement('span');
        modelTag.className = 'model-tag';
        modelTag.textContent = msg.provider_used;
        actionsBar.appendChild(modelTag);
      }

      bubble.appendChild(actionsBar);
    }

    item.appendChild(avatar);
    item.appendChild(bubble);
    elements.messagesList.appendChild(item);
  }

  async function streamMessageToDOM(msg) {
    const item = document.createElement('div');
    item.className = 'message-item assistant';
    item.id = msg.id;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"></path></svg>`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const textCard = document.createElement('div');
    textCard.className = 'msg-text-card markdown-body';
    bubble.appendChild(textCard);

    item.appendChild(avatar);
    item.appendChild(bubble);
    elements.messagesList.appendChild(item);

    const fullText = msg.content;
    const chunkLength = 16;
    let currentIdx = 0;

    while (currentIdx < fullText.length) {
      currentIdx += chunkLength;
      textCard.innerHTML = formatMarkdown(fullText.substring(0, currentIdx));
      scrollChatToBottom();
      await new Promise(r => setTimeout(r, 10));
    }

    textCard.innerHTML = formatMarkdown(fullText);
    enhanceCodeBlocks(textCard);

    // Render RAG sources if present
    if (msg.rag_sources && msg.rag_sources.length > 0) {
      const ragCard = renderRagSources(msg.rag_sources);
      if (ragCard) bubble.appendChild(ragCard);
    }

    const actionsBar = document.createElement('div');
    actionsBar.className = 'msg-actions-bar';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'action-pill-btn';
    copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> <span>Copy</span>`;
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(msg.content);
      showToast('Copied response to clipboard', 'success');
    };

    const speakBtn = document.createElement('button');
    speakBtn.className = 'action-pill-btn';
    speakBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg> <span>Listen</span>`;
    speakBtn.onclick = () => speakText(msg.content);

    actionsBar.appendChild(copyBtn);
    actionsBar.appendChild(speakBtn);

    if (msg.provider_used) {
      const modelTag = document.createElement('span');
      modelTag.className = 'model-tag';
      modelTag.textContent = msg.provider_used;
      actionsBar.appendChild(modelTag);
    }

    bubble.appendChild(actionsBar);
  }

  function enhanceCodeBlocks(container) {
    const preBlocks = container.querySelectorAll('pre');
    preBlocks.forEach(pre => {
      if (pre.parentElement.classList.contains('code-block-wrapper')) return;

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-block-header';

      const codeEl = pre.querySelector('code');
      const langMatch = codeEl ? codeEl.className.match(/language-(\w+)/) : null;
      const langName = langMatch ? langMatch[1] : 'code';

      header.innerHTML = `<span>${langName}</span>`;

      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-code-btn';
      copyBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`;
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(pre.innerText);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => {
          copyBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`;
        }, 2000);
      };

      header.appendChild(copyBtn);
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });
  }

  function scrollChatToBottom() {
    if (elements.chatContainer) {
      elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }
  }

  function speakText(text) {
    if (!window.speechSynthesis) {
      showToast('Text-to-speech not supported by browser', 'error');
      return;
    }
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[#*`_~]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.05;
    window.speechSynthesis.speak(utterance);
    showToast('Speaking response...', 'info');
  }

  /* ==========================================================================
     Lightbox & Modal Interactions
     ========================================================================== */

  function openLightbox(imgSrc) {
    state.lightboxScale = 1;
    elements.lightboxImage.src = imgSrc;
    elements.lightboxImage.style.transform = 'scale(1)';
    elements.imageLightboxModal.classList.remove('hidden');
  }

  function closeLightbox() {
    elements.imageLightboxModal.classList.add('hidden');
    elements.lightboxImage.src = '';
  }

  function openModal(modalEl) {
    modalEl.classList.remove('hidden');
  }

  function closeModal(modalEl) {
    modalEl.classList.add('hidden');
  }

  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }

  /* ==========================================================================
     Event Listeners Setup
     ========================================================================== */

  function setupEventListeners() {
    elements.themeToggleBtn.addEventListener('click', toggleTheme);

    elements.sidebarToggleBtn.addEventListener('click', () => {
      elements.sidebar.classList.toggle('collapsed');
    });
    elements.sidebarCloseBtn.addEventListener('click', () => {
      elements.sidebar.classList.add('collapsed');
    });

    elements.newChatBtn.addEventListener('click', startNewChat);

    window.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        startNewChat();
      }
    });

    elements.modelSelector.addEventListener('change', handleModelChange);

    // Pinecone RAG Toggle
    if (elements.ragToggle) {
      elements.ragToggle.addEventListener('click', () => {
        state.useRag = !state.useRag;
        elements.ragToggle.classList.toggle('active', state.useRag);
        handleModelChange();
        showToast(`Pinecone RAG ${state.useRag ? 'enabled' : 'disabled'}`, 'info');
      });
    }

    // Web Search Toggle
    elements.webSearchToggle.addEventListener('click', () => {
      state.useWebSearch = !state.useWebSearch;
      elements.webSearchToggle.classList.toggle('active', state.useWebSearch);
      showToast(`Web search ${state.useWebSearch ? 'enabled' : 'disabled'}`, 'info');
    });

    // Clear Chat
    elements.clearChatBtn.addEventListener('click', () => {
      const chat = getCurrentChat();
      if (chat) {
        chat.messages = [];
        saveConversations();
        renderCurrentChat();
        showToast('Chat cleared', 'info');
      }
    });

    // Export Chat to Markdown
    elements.exportChatBtn.addEventListener('click', () => {
      const chat = getCurrentChat();
      if (!chat || chat.messages.length === 0) {
        showToast('No messages to export', 'error');
        return;
      }
      let mdContent = `# VisionOrbit Conversation: ${chat.title}\n*Exported on ${new Date().toLocaleString()}*\n\n---\n\n`;
      chat.messages.forEach(m => {
        mdContent += `### ${m.role === 'user' ? '👤 User' : '🪐 VisionOrbit AI'}\n\n${m.content}\n\n---\n\n`;
      });
      const blob = new Blob([mdContent], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `VisionOrbit_${chat.title.replace(/[^a-zA-Z0-9]/g, '_')}.md`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Exported conversation as Markdown', 'success');
    });

    // Prompt Input Auto-resize & Keydown
    elements.promptInput.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 180) + 'px';
    });

    elements.promptInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleChatSubmit();
      }
    });

    elements.chatForm.addEventListener('submit', handleChatSubmit);

    elements.uploadBtn.addEventListener('click', () => {
      elements.imageFileInput.click();
    });

    elements.imageFileInput.addEventListener('change', (e) => {
      handleFileSelection(e.target.files);
      elements.imageFileInput.value = '';
    });

    elements.heroDropzone.addEventListener('click', () => {
      elements.imageFileInput.click();
    });

    document.querySelectorAll('.suggestion-card').forEach(card => {
      card.addEventListener('click', () => {
        const promptText = card.getAttribute('data-prompt');
        elements.promptInput.value = promptText;
        elements.promptInput.focus();
        elements.promptInput.dispatchEvent(new Event('input'));
      });
    });

    window.addEventListener('paste', (e) => {
      const items = e.clipboardData ? e.clipboardData.items : [];
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          handleFileSelection([file]);
          showToast('Image pasted from clipboard', 'info');
        }
      }
    });

    let dragCounter = 0;
    window.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      elements.dragDropOverlay.classList.add('active');
    });

    window.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        elements.dragDropOverlay.classList.remove('active');
      }
    });

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    window.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      elements.dragDropOverlay.classList.remove('active');
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files);
      }
    });

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        elements.promptInput.value += (elements.promptInput.value ? ' ' : '') + transcript;
        elements.promptInput.dispatchEvent(new Event('input'));
        elements.voiceInputBtn.classList.remove('active');
        showToast('Voice transcription captured', 'info');
      };

      recognition.onerror = () => {
        elements.voiceInputBtn.classList.remove('active');
        showToast('Speech recognition error', 'error');
      };

      recognition.onend = () => {
        elements.voiceInputBtn.classList.remove('active');
      };

      elements.voiceInputBtn.addEventListener('click', () => {
        if (elements.voiceInputBtn.classList.contains('active')) {
          recognition.stop();
          elements.voiceInputBtn.classList.remove('active');
        } else {
          recognition.start();
          elements.voiceInputBtn.classList.add('active');
          showToast('Listening... Speak now', 'info');
        }
      });
    } else {
      elements.voiceInputBtn.title = 'Speech recognition not supported in this browser';
      elements.voiceInputBtn.style.opacity = '0.5';
    }

    elements.openSettingsBtn.addEventListener('click', () => openModal(elements.settingsModal));
    elements.closeSettingsModalBtn.addEventListener('click', () => closeModal(elements.settingsModal));
    elements.settingsModal.querySelector('.modal-backdrop').addEventListener('click', () => closeModal(elements.settingsModal));
    elements.saveSettingsBtn.addEventListener('click', saveSettings);
    elements.resetSettingsBtn.addEventListener('click', resetSettings);

    elements.toggleKeyVisibility.addEventListener('click', () => {
      const isPass = elements.openaiKeyInput.type === 'password';
      elements.openaiKeyInput.type = isPass ? 'text' : 'password';
    });

    elements.lightboxCloseBtn.addEventListener('click', closeLightbox);
    elements.imageLightboxModal.querySelector('.lightbox-backdrop').addEventListener('click', closeLightbox);
    elements.lightboxZoomIn.addEventListener('click', () => {
      state.lightboxScale = Math.min(state.lightboxScale + 0.3, 3);
      elements.lightboxImage.style.transform = `scale(${state.lightboxScale})`;
    });
    elements.lightboxZoomOut.addEventListener('click', () => {
      state.lightboxScale = Math.max(state.lightboxScale - 0.3, 0.5);
      elements.lightboxImage.style.transform = `scale(${state.lightboxScale})`;
    });

    elements.chatContainer.addEventListener('scroll', () => {
      const scrollPos = elements.chatContainer.scrollTop;
      const scrollHeight = elements.chatContainer.scrollHeight;
      const clientHeight = elements.chatContainer.clientHeight;
      if (scrollHeight - scrollPos - clientHeight > 200) {
        elements.scrollToBottomBtn.classList.remove('hidden');
      } else {
        elements.scrollToBottomBtn.classList.add('hidden');
      }
    });

    elements.scrollToBottomBtn.addEventListener('click', scrollChatToBottom);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
