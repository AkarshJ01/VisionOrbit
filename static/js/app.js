/**
 * VisionOrbit - Client Application Logic
 * Modern OpenAI-style Multimodal Prompting & Pinecone RAG Intelligence Engine
 *
 * Supports:
 * - PNG / JPEG / WEBP
 * - TIFF / GeoTIFF
 * - NumPy .npy arrays
 * - YOLO-OBB detection
 * - Pinecone RAG
 * - Web search
 * - Ollama / OpenAI / VisionOrbit providers
 */

(function () {
  'use strict';

  // ==========================================================================
  // State Management
  // ==========================================================================

  const state = {
    conversations: [],
    currentChatId: null,

    // Array of staged preview images/data URLs
    stagedImages: [],

    // Metadata for staged files
    stagedFiles: [],

    isGenerating: false,

    useWebSearch: false,
    useRag: true,
    useDetection: true,

    theme: localStorage.getItem('visionorbit_theme') || 'dark',

    settings: {
      openaiKey: localStorage.getItem('visionorbit_openai_key') || '',
      ollamaUrl:
        localStorage.getItem('visionorbit_ollama_url') ||
        'http://localhost:11434',
      tavilyKey: localStorage.getItem('visionorbit_tavily_key') || '',
      systemPrompt:
        localStorage.getItem('visionorbit_sys_prompt') || ''
    },

    currentModel: 'ollama:gpt-oss:20b',

    lightboxScale: 1
  };

  // ==========================================================================
  // DOM Elements
  // ==========================================================================

  const elements = {
    appLayout: document.querySelector('.app-layout'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggleBtn: document.getElementById('sidebarToggleBtn'),
    sidebarCloseBtn: document.getElementById('sidebarCloseBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    conversationsList: document.getElementById('conversationsList'),

    currentProviderLabel:
      document.getElementById('currentProviderLabel'),
    providerSubText:
      document.getElementById('providerSubText'),
    providerDot:
      document.getElementById('providerDot'),

    themeToggleBtn:
      document.getElementById('themeToggleBtn'),
    themeIconDark:
      document.getElementById('themeIconDark'),
    themeIconLight:
      document.getElementById('themeIconLight'),

    modelSelector:
      document.getElementById('modelSelector'),

    detectionToggle:
      document.getElementById('detectionToggle'),

    ragToggle:
      document.getElementById('ragToggle'),

    webSearchToggle:
      document.getElementById('webSearchToggle'),

    exportChatBtn:
      document.getElementById('exportChatBtn'),

    clearChatBtn:
      document.getElementById('clearChatBtn'),

    chatContainer:
      document.getElementById('chatContainer'),

    welcomeHero:
      document.getElementById('welcomeHero'),

    heroDropzone:
      document.getElementById('heroDropzone'),

    messagesList:
      document.getElementById('messagesList'),

    analysisIndicator:
      document.getElementById('analysisIndicator'),

    analyzingText:
      document.getElementById('analyzingText'),

    scrollToBottomBtn:
      document.getElementById('scrollToBottomBtn'),

    chatForm:
      document.getElementById('chatForm'),

    promptInput:
      document.getElementById('promptInput'),

    imageFileInput:
      document.getElementById('imageFileInput'),

    uploadBtn:
      document.getElementById('uploadBtn'),

    voiceInputBtn:
      document.getElementById('voiceInputBtn'),

    sendBtn:
      document.getElementById('sendBtn'),

    imagePreviewTray:
      document.getElementById('imagePreviewTray'),

    dragDropOverlay:
      document.getElementById('dragDropOverlay'),

    // Settings Modal
    settingsModal:
      document.getElementById('settingsModal'),

    openSettingsBtn:
      document.getElementById('openSettingsBtn'),

    closeSettingsModalBtn:
      document.getElementById('closeSettingsModalBtn'),

    openaiKeyInput:
      document.getElementById('openaiKeyInput'),

    toggleKeyVisibility:
      document.getElementById('toggleKeyVisibility'),

    ollamaUrlInput:
      document.getElementById('ollamaUrlInput'),

    tavilyKeyInput:
      document.getElementById('tavilyKeyInput'),

    systemPromptInput:
      document.getElementById('systemPromptInput'),

    saveSettingsBtn:
      document.getElementById('saveSettingsBtn'),

    resetSettingsBtn:
      document.getElementById('resetSettingsBtn'),

    // Lightbox Modal
    imageLightboxModal:
      document.getElementById('imageLightboxModal'),

    lightboxImage:
      document.getElementById('lightboxImage'),

    lightboxCloseBtn:
      document.getElementById('lightboxCloseBtn'),

    lightboxZoomIn:
      document.getElementById('lightboxZoomIn'),

    lightboxZoomOut:
      document.getElementById('lightboxZoomOut'),

    toastContainer:
      document.getElementById('toastContainer')
  };

  // ==========================================================================
  // Configure Marked Markdown Renderer
  // ==========================================================================

  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true,

      highlight: function (code, lang) {
        if (window.hljs) {
          const language =
            hljs.getLanguage(lang)
              ? lang
              : 'plaintext';

          return hljs.highlight(code, {
            language
          }).value;
        }

        return code;
      }
    });
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

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

  // ==========================================================================
  // Theme
  // ==========================================================================

  function loadTheme() {
    document.documentElement.setAttribute(
      'data-theme',
      state.theme
    );

    if (state.theme === 'light') {
      elements.themeIconDark.classList.add('hidden');
      elements.themeIconLight.classList.remove('hidden');
    } else {
      elements.themeIconDark.classList.remove('hidden');
      elements.themeIconLight.classList.add('hidden');
    }
  }

  function toggleTheme() {
    state.theme =
      state.theme === 'dark'
        ? 'light'
        : 'dark';

    localStorage.setItem(
      'visionorbit_theme',
      state.theme
    );

    loadTheme();

    showToast(
      `Switched to ${state.theme} mode`,
      'info'
    );
  }

  // ==========================================================================
  // Settings
  // ==========================================================================

  function loadSettings() {
    elements.openaiKeyInput.value =
      state.settings.openaiKey;

    elements.ollamaUrlInput.value =
      state.settings.ollamaUrl;

    elements.tavilyKeyInput.value =
      state.settings.tavilyKey;

    elements.systemPromptInput.value =
      state.settings.systemPrompt;
  }

  function saveSettings() {
    state.settings.openaiKey =
      elements.openaiKeyInput.value.trim();

    state.settings.ollamaUrl =
      elements.ollamaUrlInput.value.trim() ||
      'http://localhost:11434';

    state.settings.tavilyKey =
      elements.tavilyKeyInput.value.trim();

    state.settings.systemPrompt =
      elements.systemPromptInput.value.trim();

    localStorage.setItem(
      'visionorbit_openai_key',
      state.settings.openaiKey
    );

    localStorage.setItem(
      'visionorbit_ollama_url',
      state.settings.ollamaUrl
    );

    localStorage.setItem(
      'visionorbit_tavily_key',
      state.settings.tavilyKey
    );

    localStorage.setItem(
      'visionorbit_sys_prompt',
      state.settings.systemPrompt
    );

    closeModal(elements.settingsModal);

    showToast(
      'Settings saved successfully',
      'success'
    );

    fetchModelCapabilities();
  }

  function resetSettings() {
    elements.openaiKeyInput.value = '';
    elements.ollamaUrlInput.value =
      'http://localhost:11434';
    elements.tavilyKeyInput.value = '';
    elements.systemPromptInput.value = '';

    saveSettings();
  }

  // ==========================================================================
  // Model Capabilities
  // ==========================================================================

  async function fetchModelCapabilities() {
    try {
      const url =
        `/api/models?ollama_url=${encodeURIComponent(
          state.settings.ollamaUrl
        )}`;

      const res = await fetch(url);

      if (res.ok) {
        const data = await res.json();

        updateModelDropdown(
          data.providers
        );
      }
    } catch (e) {
      console.warn(
        'Could not refresh model capabilities:',
        e
      );
    }
  }

  function updateModelDropdown(providers) {
    if (!elements.modelSelector) return;

    const currentVal =
      elements.modelSelector.value;

    elements.modelSelector.innerHTML = '';

    providers.forEach(p => {
      const optgroup =
        document.createElement('optgroup');

      optgroup.label =
        `${p.name} (${p.badge})`;

      p.models.forEach(m => {
        const option =
          document.createElement('option');

        option.value =
          `${p.id}:${m.id}`;

        option.textContent =
          m.name;

        optgroup.appendChild(option);
      });

      elements.modelSelector.appendChild(
        optgroup
      );
    });

    if (
      currentVal &&
      elements.modelSelector.querySelector(
        `option[value="${currentVal}"]`
      )
    ) {
      elements.modelSelector.value =
        currentVal;
    } else if (
      elements.modelSelector.querySelector(
        'option[value="ollama:gpt-oss:20b"]'
      )
    ) {
      elements.modelSelector.value =
        'ollama:gpt-oss:20b';
    } else {
      elements.modelSelector.value =
        'builtin:visionorbit-core';
    }

    handleModelChange();
  }

  function handleModelChange() {
    const val =
      elements.modelSelector.value;

    state.currentModel = val;

    const [provider, model] =
      val.split(':');

    if (provider === 'ollama') {
      elements.currentProviderLabel.textContent =
        `Ollama (${model})`;

      elements.providerSubText.textContent =
        state.useRag
          ? 'Pinecone RAG Active'
          : 'Local LLM Inference';

      elements.providerDot.className =
        'provider-dot active';

    } else if (provider === 'openai') {
      elements.currentProviderLabel.textContent =
        `OpenAI (${model})`;

      elements.providerSubText.textContent =
        state.settings.openaiKey
          ? 'Live Multimodal Inference'
          : 'Using Server/Env Key';

      elements.providerDot.className =
        'provider-dot active';

    } else {
      elements.currentProviderLabel.textContent =
        'VisionOrbit Smart Core';

      elements.providerSubText.textContent =
        'Instant Multimodal Analysis';

      elements.providerDot.className =
        'provider-dot active';
    }
  }

  // ==========================================================================
  // Conversation History
  // ==========================================================================

  function loadConversations() {
    try {
      const saved =
        localStorage.getItem(
          'visionorbit_conversations'
        );

      state.conversations =
        saved
          ? JSON.parse(saved)
          : [];

    } catch (e) {
      state.conversations = [];
    }

    if (state.conversations.length > 0) {
      switchConversation(
        state.conversations[0].id
      );
    } else {
      startNewChat();
    }

    renderConversationsList();
  }

  function saveConversations() {
    localStorage.setItem(
      'visionorbit_conversations',
      JSON.stringify(state.conversations)
    );

    renderConversationsList();
  }

  function startNewChat() {
    const newChat = {
      id: 'chat_' + Date.now(),
      title: 'New Inquiry',
      messages: [],
      createdAt: Date.now()
    };

    state.conversations.unshift(
      newChat
    );

    state.currentChatId =
      newChat.id;

    state.stagedImages = [];
    state.stagedFiles = [];

    renderStagedImages();

    saveConversations();
    renderCurrentChat();
  }

  function switchConversation(chatId) {
    state.currentChatId =
      chatId;

    state.stagedImages = [];
    state.stagedFiles = [];

    renderStagedImages();
    renderConversationsList();
    renderCurrentChat();
  }

  function deleteConversation(chatId, e) {
    if (e) {
      e.stopPropagation();
    }

    state.conversations =
      state.conversations.filter(
        c => c.id !== chatId
      );

    if (
      state.currentChatId ===
      chatId
    ) {
      if (
        state.conversations.length > 0
      ) {
        state.currentChatId =
          state.conversations[0].id;
      } else {
        startNewChat();
        return;
      }
    }

    saveConversations();
    renderCurrentChat();
  }

  function renderConversationsList() {
    if (!elements.conversationsList) {
      return;
    }

    elements.conversationsList.innerHTML =
      '';

    state.conversations.forEach(
      chat => {
        const item =
          document.createElement('div');

        item.className =
          `chat-history-item ${
            chat.id === state.currentChatId
              ? 'active'
              : ''
          }`;

        item.onclick = () =>
          switchConversation(
            chat.id
          );

        const titleSpan =
          document.createElement('span');

        titleSpan.className =
          'history-item-title';

        titleSpan.textContent =
          chat.title ||
          'New Inquiry';

        const delBtn =
          document.createElement('button');

        delBtn.className =
          'history-item-delete';

        delBtn.title =
          'Delete conversation';

        delBtn.innerHTML = `
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        `;

        delBtn.onclick =
          e =>
            deleteConversation(
              chat.id,
              e
            );

        item.appendChild(
          titleSpan
        );

        item.appendChild(
          delBtn
        );

        elements.conversationsList.appendChild(
          item
        );
      }
    );
  }

  function getCurrentChat() {
    return state.conversations.find(
      c =>
        c.id ===
        state.currentChatId
    );
  }

  function renderCurrentChat() {
    const chat =
      getCurrentChat();

    if (
      !chat ||
      chat.messages.length === 0
    ) {
      elements.welcomeHero.classList.remove(
        'hidden'
      );

      elements.messagesList.classList.add(
        'hidden'
      );

      elements.messagesList.innerHTML =
        '';

      return;
    }

    elements.welcomeHero.classList.add(
      'hidden'
    );

    elements.messagesList.classList.remove(
      'hidden'
    );

    elements.messagesList.innerHTML =
      '';

    chat.messages.forEach(
      msg => {
        appendMessageToDOM(msg);
      }
    );

    scrollChatToBottom();
  }

  // ==========================================================================
  // File Upload & Staging
  // ==========================================================================

  async function handleFileSelection(files) {
    if (!files || files.length === 0) {
      return;
    }

    for (
      const file of Array.from(files)
    ) {
      const nameLower =
        file.name.toLowerCase();

      // --------------------------------------------------------------
      // Supported image formats
      // --------------------------------------------------------------

      const isImage =
        file.type.startsWith('image/') ||
        nameLower.endsWith('.tif') ||
        nameLower.endsWith('.tiff');

      // --------------------------------------------------------------
      // NumPy format
      // --------------------------------------------------------------

      const isNpy =
        nameLower.endsWith('.npy');

      // --------------------------------------------------------------
      // Validate extension
      // --------------------------------------------------------------

      if (!isImage && !isNpy) {
        showToast(
          `Unsupported file type: ${file.name}`,
          'error'
        );

        continue;
      }

      // --------------------------------------------------------------
      // File size limit
      // --------------------------------------------------------------

      if (
        file.size >
        100 * 1024 * 1024
      ) {
        showToast(
          `File too large (max 100MB): ${file.name}`,
          'error'
        );

        continue;
      }

      try {
        const formData =
          new FormData();

        formData.append(
          'file',
          file
        );

        // ------------------------------------------------------------
        // Upload to backend
        // ------------------------------------------------------------

        const res =
          await fetch(
            '/api/upload',
            {
              method: 'POST',
              body: formData
            }
          );

        if (!res.ok) {
          const errorData =
            await res
              .json()
              .catch(() => ({}));

          throw new Error(
            errorData.detail ||
            `Upload failed with status ${res.status}`
          );
        }

        const uploadData =
          await res.json();

        // ------------------------------------------------------------
// Store uploaded file reference
// ------------------------------------------------------------

if (isNpy) {

  if (!uploadData.npy_reference) {
    throw new Error(
      'Server did not return an NPY reference'
    );
  }

  state.stagedImages.push(
    uploadData.npy_reference
  );

} else {

  if (!uploadData.data_url) {
    throw new Error(
      'Server did not return a preview image'
    );
  }

  state.stagedImages.push(
    uploadData.data_url
  );
}

        // ------------------------------------------------------------
        // Store preview image
        // ------------------------------------------------------------

        

        // ------------------------------------------------------------
        // Store file metadata
        // ------------------------------------------------------------

        state.stagedFiles.push({
          filename:
            uploadData.filename ||
            file.name,

          file_type:
            uploadData.file_type ||
            (isNpy
              ? 'npy'
              : 'image'),

          mime_type:
            uploadData.mime_type ||
            file.type,

          metadata:
            uploadData.metadata ||
            {}
        });

        renderStagedImages();

        // ------------------------------------------------------------
        // NPY-specific toast
        // ------------------------------------------------------------

        if (isNpy) {
          const metadata =
            uploadData.metadata ||
            {};

          const shape =
            metadata.shape ||
            metadata.original_shape ||
            'unknown shape';

          const dtype =
            metadata.dtype ||
            'unknown dtype';

          showToast(
            `Loaded NumPy array: ${file.name} — ${shape} — ${dtype}`,
            'success'
          );

        } else {
          // ----------------------------------------------------------
          // Normal image toast
          // ----------------------------------------------------------

          showToast(
            `Loaded satellite visual: ${file.name}`,
            'success'
          );
        }

      } catch (err) {
        console.warn(
          'Server upload failed:',
          err
        );

        // ------------------------------------------------------------
        // IMPORTANT:
        //
        // Only images get the FileReader fallback.
        //
        // .npy cannot be sent to FileReader as an image.
        // ------------------------------------------------------------

        if (isImage) {
          const reader =
            new FileReader();

          reader.onload =
            e => {
              state.stagedImages.push(
                e.target.result
              );

              state.stagedFiles.push({
                filename: file.name,
                file_type: 'image',
                mime_type: file.type,
                metadata: {}
              });

              renderStagedImages();

              showToast(
                `Attached image: ${file.name}`,
                'info'
              );
            };

          reader.readAsDataURL(
            file
          );

        } else {
          showToast(
            `Could not process NumPy file: ${file.name}`,
            'error'
          );
        }
      }
    }
  }

  // ==========================================================================
  // Render Staged Images / NPY Files
  // ==========================================================================

  function renderStagedImages() {
    if (!elements.imagePreviewTray) {
      return;
    }

    if (
      state.stagedImages.length === 0
    ) {
      elements.imagePreviewTray.classList.add(
        'hidden'
      );

      elements.imagePreviewTray.innerHTML =
        '';

      return;
    }

    elements.imagePreviewTray.classList.remove(
      'hidden'
    );

    elements.imagePreviewTray.innerHTML =
      '';

    state.stagedImages.forEach(
      (imgData, index) => {
        const card =
          document.createElement('div');

        card.className =
          'staged-img-card';

        // ------------------------------------------------------------
        // Preview image
        // ------------------------------------------------------------

        const img =
          document.createElement('img');

        img.src =
          imgData;

        img.alt =
          `Staged visual ${index + 1}`;

        img.onclick =
          () =>
            openLightbox(
              imgData
            );

        card.appendChild(img);

        // ------------------------------------------------------------
        // File information
        // ------------------------------------------------------------

        const fileInfo =
          state.stagedFiles[index];

        if (fileInfo) {
          const info =
            document.createElement('div');

          info.className =
            'staged-file-info';

          if (
            fileInfo.file_type ===
            'npy'
          ) {
            const metadata =
              fileInfo.metadata ||
              {};

            const shape =
              metadata.shape ||
              metadata.original_shape ||
              '';

            const dtype =
              metadata.dtype ||
              '';

            info.textContent =
              `NPY${shape ? ` • ${shape}` : ''}${dtype ? ` • ${dtype}` : ''}`;

          } else {
            info.textContent =
              fileInfo.filename ||
              '';
          }

          card.appendChild(info);
        }

        // ------------------------------------------------------------
        // Remove button
        // ------------------------------------------------------------

        const removeBtn =
          document.createElement(
            'button'
          );

        removeBtn.className =
          'staged-img-remove';

        removeBtn.innerHTML =
          '&times;';

        removeBtn.title =
          'Remove attachment';

        removeBtn.onclick =
          e => {
            e.stopPropagation();

            state.stagedImages.splice(
              index,
              1
            );

            state.stagedFiles.splice(
              index,
              1
            );

            renderStagedImages();
          };

        card.appendChild(
          removeBtn
        );

        elements.imagePreviewTray.appendChild(
          card
        );
      }
    );
  }

  // ==========================================================================
  // Chat Submission
  // ==========================================================================

  async function handleChatSubmit(e) {
    if (e) {
      e.preventDefault();
    }

    if (state.isGenerating) {
      return;
    }

    const prompt =
      elements.promptInput.value.trim();

    const images =
      [...state.stagedImages];

    const stagedFiles =
      [...state.stagedFiles];

    if (
      !prompt &&
      images.length === 0
    ) {
      showToast(
        'Please enter a query or attach an image/NPY file.',
        'error'
      );

      return;
    }

    elements.promptInput.value =
      '';

    elements.promptInput.style.height =
      'auto';

    state.stagedImages = [];
    state.stagedFiles = [];

    renderStagedImages();

    const chat =
      getCurrentChat();

    if (!chat) {
      return;
    }

    // --------------------------------------------------------------
    // Create user message
    // --------------------------------------------------------------

    const userMsg = {
      id: 'msg_' + Date.now(),

      role: 'user',

      content:
        prompt ||
        'Analyze and describe the attached image(s) or NumPy array(s).',

      images: images,

      files: stagedFiles,

      timestamp: Date.now()
    };

    // --------------------------------------------------------------
    // Automatic conversation title
    // --------------------------------------------------------------

    if (
      chat.messages.length === 0
    ) {
      const autoTitle =
        prompt
          ? (
              prompt.length > 32
                ? prompt.substring(
                    0,
                    32
                  ) + '...'
                : prompt
            )
          : 'Visual & Target Analysis';

      chat.title =
        autoTitle;

      saveConversations();
    }

    chat.messages.push(
      userMsg
    );

    saveConversations();

    elements.welcomeHero.classList.add(
      'hidden'
    );

    elements.messagesList.classList.remove(
      'hidden'
    );

    appendMessageToDOM(
      userMsg
    );

    scrollChatToBottom();

    await generateAssistantResponse(
      prompt,
      images,
      stagedFiles
    );
  }

  // ==========================================================================
  // Assistant Response
  // ==========================================================================

  async function generateAssistantResponse(
    prompt,
    images,
    stagedFiles = []
  ) {
    state.isGenerating = true;

    updateSendButtonState();

    elements.analysisIndicator.classList.remove(
      'hidden'
    );

    elements.analyzingText.textContent =
      state.useDetection &&
      images.length > 0
        ? 'Running YOLO-OBB detection & retrieving Pinecone knowledge...'
        : (
            state.useRag
              ? 'Retrieving from Pinecone knowledge base & thinking...'
              : 'Generating response...'
          );

    scrollChatToBottom();

    const [
      provider,
      model
    ] =
      state.currentModel.split(':');

    const chat =
      getCurrentChat();

    // --------------------------------------------------------------
    // History payload
    // --------------------------------------------------------------

    const historyPayload =
      chat.messages
        .slice(0, -1)
        .map(m => ({
          role: m.role,
          content: m.content,

          detection_results:
            m.detection_results ||
            null,

          images:
            m.images ||
            [],

          files:
            m.files ||
            []
        }));

    try {
      // ------------------------------------------------------------
      // Main API payload
      // ------------------------------------------------------------

      const payload = {
        prompt:
          prompt ||
          'Describe and analyze the attached satellite image or NumPy array in detail.',

        images:
          images,

        // NPY metadata is included here for future backend use.
        files:
          stagedFiles,

        history:
          historyPayload,

        provider:
          provider,

        model:
          model,

        apiKey:
          state.settings.openaiKey ||
          null,

        ollamaBaseUrl:
          state.settings.ollamaUrl ||
          null,

        tavilyApiKey:
          state.settings.tavilyKey ||
          null,

        useWebSearch:
          state.useWebSearch,

        useRag:
          state.useRag,

        useDetection:
          state.useDetection,

        systemPrompt:
          state.settings.systemPrompt ||
          null
      };

      // ------------------------------------------------------------
      // Chat API request
      // ------------------------------------------------------------

      const response =
        await fetch(
          '/api/chat',
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json'
            },

            body:
              JSON.stringify(
                payload
              )
          }
        );

      if (!response.ok) {
        const errData =
          await response
            .json()
            .catch(() => ({}));

        throw new Error(
          errData.detail ||
          `Server error (${response.status})`
        );
      }

      const data =
        await response.json();

      // ------------------------------------------------------------
      // Create assistant message
      // ------------------------------------------------------------

      const assistantMsg = {
        id:
          'msg_' +
          Date.now(),

        role:
          'assistant',

        content:
          data.reply,

        provider_used:
          data.provider_used,

        model_used:
          data.model_used,

        annotated_image:
          data.annotated_image ||
          null,

        detection_results:
          data.detection_results ||
          null,

        image_metadata:
          data.image_metadata ||
          [],

        search_sources:
          data.search_sources ||
          [],

        rag_sources:
          data.rag_sources ||
          [],

        timestamp:
          Date.now()
      };

      chat.messages.push(
        assistantMsg
      );

      saveConversations();

      elements.analysisIndicator.classList.add(
        'hidden'
      );

      await streamMessageToDOM(
        assistantMsg
      );

    } catch (err) {
      console.error(
        'Chat error:',
        err
      );

      elements.analysisIndicator.classList.add(
        'hidden'
      );

      const errorMsg = {
        id:
          'msg_' +
          Date.now(),

        role:
          'assistant',

        content:
          `⚠️ **An error occurred during generation:**\n\n\`${err.message}\`\n\n*Please verify your Pinecone index or model settings in the ⚙️ Settings dialog.*`,

        provider_used:
          'Error Handler',

        timestamp:
          Date.now()
      };

      chat.messages.push(
        errorMsg
      );

      saveConversations();

      appendMessageToDOM(
        errorMsg
      );

    } finally {
      state.isGenerating =
        false;

      updateSendButtonState();

      scrollChatToBottom();
    }
  }

  // ==========================================================================
  // Send Button
  // ==========================================================================

  function updateSendButtonState() {
    if (
      state.isGenerating
    ) {
      elements.sendBtn.disabled =
        true;

      elements.sendBtn.innerHTML =
        `<span class="pulse-spark"></span>`;

    } else {
      elements.sendBtn.disabled =
        false;

      elements.sendBtn.innerHTML =
        `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
        >
          <line
            x1="12"
            y1="19"
            x2="12"
            y2="5"
          ></line>

          <polyline
            points="5 12 12 5 19 12"
          ></polyline>
        </svg>
        `;
    }
  }

  // ==========================================================================
  // Detection Card
  // ==========================================================================

  function renderDetectionCard(msg) {
    if (
      !msg.detection_results &&
      !msg.annotated_image
    ) {
      return null;
    }

    const det =
      msg.detection_results ||
      {};

    const totalDets =
      det.total_detections !==
      undefined
        ? det.total_detections
        : (
            det.detections
              ? det.detections.length
              : 0
          );

    const classCounts =
      det.class_counts ||
      {};

    const card =
      document.createElement(
        'div'
      );

    card.className =
      'detection-telemetry-card';

    // --------------------------------------------------------------
    // Header
    // --------------------------------------------------------------

    const header =
      document.createElement(
        'div'
      );

    header.className =
      'detection-card-header';

    header.innerHTML = `
      <div class="det-header-left">
        <span class="det-icon">🎯</span>

        <div>
          <div class="det-title">
            YOLO-OBB Detection Telemetry
          </div>

          <div class="det-subtitle">
            ${totalDets}
            Target${totalDets === 1 ? '' : 's'}
            Identified •
            ${det.resolution || 'High-Res'} •
            ${det.crs || 'WGS84'}
          </div>
        </div>
      </div>

      <div class="det-header-right">
        <span class="det-count-badge">
          ${totalDets} Detections
        </span>
      </div>
    `;

    card.appendChild(
      header
    );

    // --------------------------------------------------------------
    // Class counts
    // --------------------------------------------------------------

    if (
      Object.keys(
        classCounts
      ).length > 0
    ) {
      const chipsBar =
        document.createElement(
          'div'
        );

      chipsBar.className =
        'det-chips-bar';

      Object.entries(
        classCounts
      ).forEach(
        ([cls, count]) => {
          const pill =
            document.createElement(
              'span'
            );

          pill.className =
            'det-pill';

          pill.innerHTML =
            `<strong>${count}</strong> ${cls}`;

          chipsBar.appendChild(
            pill
          );
        }
      );

      card.appendChild(
        chipsBar
      );
    }

    // --------------------------------------------------------------
    // Annotated image
    // --------------------------------------------------------------

    if (
      msg.annotated_image
    ) {
      const visualContainer =
        document.createElement(
          'div'
        );

      visualContainer.className =
        'det-visual-container';

      const imgWrapper =
        document.createElement(
          'div'
        );

      imgWrapper.className =
        'det-img-wrapper';

      const img =
        document.createElement(
          'img'
        );

      img.src =
        msg.annotated_image;

      img.alt =
        'YOLO Oriented Bounding Box Detections';

      img.className =
        'det-annotated-image';

      img.onclick =
        () =>
          openLightbox(
            msg.annotated_image
          );

      const overlayBadge =
        document.createElement(
          'div'
        );

      overlayBadge.className =
        'det-img-badge';

      overlayBadge.innerHTML =
        `<span>🔍 Click for Full-Screen Inspection</span>`;

      imgWrapper.appendChild(
        img
      );

      imgWrapper.appendChild(
        overlayBadge
      );

      visualContainer.appendChild(
        imgWrapper
      );

      card.appendChild(
        visualContainer
      );
    }

    return card;
  }

  // ==========================================================================
  // Markdown
  // ==========================================================================

  function formatMarkdown(
    rawText
  ) {
    if (window.marked) {
      return marked.parse(
        rawText
      );
    }

    return rawText
      .replace(
        /\n\n/g,
        '<p></p>'
      )
      .replace(
        /\n/g,
        '<br>'
      )
      .replace(
        /\*\*(.*?)\*\*/g,
        '<strong>$1</strong>'
      )
      .replace(
        /\*(.*?)\*/g,
        '<em>$1</em>'
      );
  }

  // ==========================================================================
  // RAG Sources
  // ==========================================================================

  function renderRagSources(
    sources
  ) {
    if (
      !sources ||
      sources.length === 0
    ) {
      return null;
    }

    const container =
      document.createElement(
        'div'
      );

    container.className =
      'rag-sources-card';

    const header =
      document.createElement(
        'div'
      );

    header.className =
      'rag-sources-header';

    header.innerHTML = `
      <span>
        📚 Pinecone Knowledge Sources
        (${sources.length} chunks retrieved)
      </span>

      <span class="accordion-toggle">
        ▾
      </span>
    `;

    const list =
      document.createElement(
        'div'
      );

    list.className =
      'rag-sources-list';

    sources.forEach(
      s => {
        const item =
          document.createElement(
            'div'
          );

        item.className =
          'rag-source-item';

        item.innerHTML = `
          <div class="rag-source-meta">
            <span class="rag-source-filename">
              📄 ${s.filename}
            </span>

            <span class="rag-source-page">
              Page ${s.page}
            </span>
          </div>

          <div class="rag-source-snippet">
            ${s.snippet}
          </div>
        `;

        list.appendChild(
          item
        );
      }
    );

    header.onclick = () => {
      list.classList.toggle(
        'hidden'
      );

      const toggleIcon =
        header.querySelector(
          '.accordion-toggle'
        );

      if (toggleIcon) {
        toggleIcon.textContent =
          list.classList.contains(
            'hidden'
          )
            ? '▸'
            : '▾';
      }
    };

    container.appendChild(
      header
    );

    container.appendChild(
      list
    );

    return container;
  }

  // ==========================================================================
  // Message Rendering
  // ==========================================================================

  function appendMessageToDOM(
    msg
  ) {
    const item =
      document.createElement(
        'div'
      );

    item.className =
      `message-item ${msg.role}`;

    item.id =
      msg.id;

    const avatar =
      document.createElement(
        'div'
      );

    avatar.className =
      'msg-avatar';

    if (
      msg.role === 'user'
    ) {
      avatar.textContent =
        'U';
    } else {
      avatar.innerHTML = `
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <circle
            cx="12"
            cy="12"
            r="3"
          ></circle>

          <path
            d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"
          ></path>
        </svg>
      `;
    }

    const bubble =
      document.createElement(
        'div'
      );

    bubble.className =
      'msg-bubble';

    // --------------------------------------------------------------
    // Uploaded images / NPY previews
    // --------------------------------------------------------------

    if (
      msg.images &&
      msg.images.length > 0
    ) {
      const imgStrip =
        document.createElement(
          'div'
        );

      imgStrip.className =
        'msg-images-strip';

      msg.images.forEach(
        (imgData, i) => {
          const thumbWrap =
            document.createElement(
              'div'
            );

          thumbWrap.className =
            'msg-thumb-wrapper';

          thumbWrap.onclick =
            () =>
              openLightbox(
                imgData
              );

          const img =
            document.createElement(
              'img'
            );

          img.src =
            imgData;

          img.alt =
            `Uploaded visual ${i + 1}`;

          const badge =
            document.createElement(
              'span'
            );

          badge.className =
            'thumb-zoom-badge';

          // Show NPY badge when possible
          const file =
            msg.files &&
            msg.files[i];

          if (
            file &&
            file.file_type ===
              'npy'
          ) {
            badge.textContent =
              '🔢 NPY • Click to Zoom';
          } else {
            badge.textContent =
              '🔍 Click to Zoom';
          }

          thumbWrap.appendChild(
            img
          );

          thumbWrap.appendChild(
            badge
          );

          imgStrip.appendChild(
            thumbWrap
          );
        }
      );

      bubble.appendChild(
        imgStrip
      );
    }

    // --------------------------------------------------------------
    // YOLO Detection Card
    // --------------------------------------------------------------

    if (
      msg.detection_results ||
      msg.annotated_image
    ) {
      const detCard =
        renderDetectionCard(
          msg
        );

      if (detCard) {
        bubble.appendChild(
          detCard
        );
      }
    }

    // --------------------------------------------------------------
    // Text
    // --------------------------------------------------------------

    const textCard =
      document.createElement(
        'div'
      );

    textCard.className =
      'msg-text-card markdown-body';

    textCard.innerHTML =
      formatMarkdown(
        msg.content
      );

    enhanceCodeBlocks(
      textCard
    );

    bubble.appendChild(
      textCard
    );

    // --------------------------------------------------------------
    // RAG sources
    // --------------------------------------------------------------

    if (
      msg.rag_sources &&
      msg.rag_sources.length > 0
    ) {
      const ragCard =
        renderRagSources(
          msg.rag_sources
        );

      if (ragCard) {
        bubble.appendChild(
          ragCard
        );
      }
    }

    // --------------------------------------------------------------
    // Assistant actions
    // --------------------------------------------------------------

    if (
      msg.role === 'assistant'
    ) {
      const actionsBar =
        document.createElement(
          'div'
        );

      actionsBar.className =
        'msg-actions-bar';

      // Copy
      const copyBtn =
        document.createElement(
          'button'
        );

      copyBtn.className =
        'action-pill-btn';

      copyBtn.innerHTML = `
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <rect
            x="9"
            y="9"
            width="13"
            height="13"
            rx="2"
          ></rect>

          <path
            d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"
          ></path>
        </svg>

        <span>Copy</span>
      `;

      copyBtn.onclick =
        () => {
          navigator.clipboard.writeText(
            msg.content
          );

          showToast(
            'Copied response to clipboard',
            'success'
          );
        };

      // Listen
      const speakBtn =
        document.createElement(
          'button'
        );

      speakBtn.className =
        'action-pill-btn';

      speakBtn.innerHTML = `
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <polygon
            points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"
          ></polygon>

          <path
            d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"
          ></path>
        </svg>

        <span>Listen</span>
      `;

      speakBtn.onclick =
        () =>
          speakText(
            msg.content
          );

      actionsBar.appendChild(
        copyBtn
      );

      actionsBar.appendChild(
        speakBtn
      );

      if (
        msg.provider_used
      ) {
        const modelTag =
          document.createElement(
            'span'
          );

        modelTag.className =
          'model-tag';

        modelTag.textContent =
          msg.provider_used;

        actionsBar.appendChild(
          modelTag
        );
      }

      bubble.appendChild(
        actionsBar
      );
    }

    item.appendChild(
      avatar
    );

    item.appendChild(
      bubble
    );

    elements.messagesList.appendChild(
      item
    );
  }

  // ==========================================================================
  // Streaming Assistant Message
  // ==========================================================================

  async function streamMessageToDOM(
    msg
  ) {
    const item =
      document.createElement(
        'div'
      );

    item.className =
      'message-item assistant';

    item.id =
      msg.id;

    const avatar =
      document.createElement(
        'div'
      );

    avatar.className =
      'msg-avatar';

    avatar.innerHTML = `
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <circle
          cx="12"
          cy="12"
          r="3"
        ></circle>

        <path
          d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0"
        ></path>
      </svg>
    `;

    const bubble =
      document.createElement(
        'div'
      );

    bubble.className =
      'msg-bubble';

    // --------------------------------------------------------------
    // Detection
    // --------------------------------------------------------------

    if (
      msg.detection_results ||
      msg.annotated_image
    ) {
      const detCard =
        renderDetectionCard(
          msg
        );

      if (detCard) {
        bubble.appendChild(
          detCard
        );
      }
    }

    // --------------------------------------------------------------
    // Text
    // --------------------------------------------------------------

    const textCard =
      document.createElement(
        'div'
      );

    textCard.className =
      'msg-text-card markdown-body';

    bubble.appendChild(
      textCard
    );

    item.appendChild(
      avatar
    );

    item.appendChild(
      bubble
    );

    elements.messagesList.appendChild(
      item
    );

    // --------------------------------------------------------------
    // Simulated streaming
    // --------------------------------------------------------------

    const fullText =
      msg.content;

    const chunkLength =
      16;

    let currentIdx =
      0;

    while (
      currentIdx <
      fullText.length
    ) {
      currentIdx +=
        chunkLength;

      textCard.innerHTML =
        formatMarkdown(
          fullText.substring(
            0,
            currentIdx
          )
        );

      scrollChatToBottom();

      await new Promise(
        r =>
          setTimeout(
            r,
            10
          )
      );
    }

    textCard.innerHTML =
      formatMarkdown(
        fullText
      );

    enhanceCodeBlocks(
      textCard
    );

    // --------------------------------------------------------------
    // RAG sources
    // --------------------------------------------------------------

    if (
      msg.rag_sources &&
      msg.rag_sources.length > 0
    ) {
      const ragCard =
        renderRagSources(
          msg.rag_sources
        );

      if (ragCard) {
        bubble.appendChild(
          ragCard
        );
      }
    }

    // --------------------------------------------------------------
    // Actions
    // --------------------------------------------------------------

    const actionsBar =
      document.createElement(
        'div'
      );

    actionsBar.className =
      'msg-actions-bar';

    // Copy
    const copyBtn =
      document.createElement(
        'button'
      );

    copyBtn.className =
      'action-pill-btn';

    copyBtn.innerHTML = `
      <svg
        width="13"
        height="13"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <rect
          x="9"
          y="9"
          width="13"
          height="13"
          rx="2"
        ></rect>

        <path
          d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"
        ></path>
      </svg>

      <span>Copy</span>
    `;

    copyBtn.onclick =
      () => {
        navigator.clipboard.writeText(
          msg.content
        );

        showToast(
          'Copied response to clipboard',
          'success'
        );
      };

    // Listen
    const speakBtn =
      document.createElement(
        'button'
      );

    speakBtn.className =
      'action-pill-btn';

    speakBtn.innerHTML = `
      <svg
        width="13"
        height="13"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <polygon
          points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"
        ></polygon>

        <path
          d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"
        ></path>
      </svg>

      <span>Listen</span>
    `;

    speakBtn.onclick =
      () =>
        speakText(
          msg.content
        );

    actionsBar.appendChild(
      copyBtn
    );

    actionsBar.appendChild(
      speakBtn
    );

    if (
      msg.provider_used
    ) {
      const modelTag =
        document.createElement(
          'span'
        );

      modelTag.className =
        'model-tag';

      modelTag.textContent =
        msg.provider_used;

      actionsBar.appendChild(
        modelTag
      );
    }

    bubble.appendChild(
      actionsBar
    );
  }

  // ==========================================================================
  // Code Blocks
  // ==========================================================================

  function enhanceCodeBlocks(
    container
  ) {
    const preBlocks =
      container.querySelectorAll(
        'pre'
      );

    preBlocks.forEach(
      pre => {
        if (
          pre.parentElement.classList.contains(
            'code-block-wrapper'
          )
        ) {
          return;
        }

        const wrapper =
          document.createElement(
            'div'
          );

        wrapper.className =
          'code-block-wrapper';

        const header =
          document.createElement(
            'div'
          );

        header.className =
          'code-block-header';

        const codeEl =
          pre.querySelector(
            'code'
          );

        const langMatch =
          codeEl
            ? codeEl.className.match(
                /language-(\w+)/
              )
            : null;

        const langName =
          langMatch
            ? langMatch[1]
            : 'code';

        header.innerHTML =
          `<span>${langName}</span>`;

        const copyBtn =
          document.createElement(
            'button'
          );

        copyBtn.className =
          'copy-code-btn';

        copyBtn.innerHTML = `
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <rect
              x="9"
              y="9"
              width="13"
              height="13"
              rx="2"
            ></rect>

            <path
              d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"
            ></path>
          </svg>

          Copy
        `;

        copyBtn.onclick =
          () => {
            navigator.clipboard.writeText(
              pre.innerText
            );

            copyBtn.textContent =
              'Copied!';

            setTimeout(
              () => {
                copyBtn.innerHTML = `
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                  >
                    <rect
                      x="9"
                      y="9"
                      width="13"
                      height="13"
                      rx="2"
                    ></rect>

                    <path
                      d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"
                    ></path>
                  </svg>

                  Copy
                `;
              },
              2000
            );
          };

        header.appendChild(
          copyBtn
        );

        pre.parentNode.insertBefore(
          wrapper,
          pre
        );

        wrapper.appendChild(
          header
        );

        wrapper.appendChild(
          pre
        );
      }
    );
  }

  // ==========================================================================
  // Scrolling
  // ==========================================================================

  function scrollChatToBottom() {
    if (
      elements.chatContainer
    ) {
      elements.chatContainer.scrollTop =
        elements.chatContainer.scrollHeight;
    }
  }

  // ==========================================================================
  // Text To Speech
  // ==========================================================================

  function speakText(
    text
  ) {
    if (
      !window.speechSynthesis
    ) {
      showToast(
        'Text-to-speech not supported by browser',
        'error'
      );

      return;
    }

    window.speechSynthesis.cancel();

    const cleanText =
      text.replace(
        /[#*`_~]/g,
        ''
      );

    const utterance =
      new SpeechSynthesisUtterance(
        cleanText
      );

    utterance.rate =
      1.05;

    window.speechSynthesis.speak(
      utterance
    );

    showToast(
      'Speaking response...',
      'info'
    );
  }

  // ==========================================================================
  // Lightbox
  // ==========================================================================

  function openLightbox(
    imgSrc
  ) {
    state.lightboxScale =
      1;

    elements.lightboxImage.src =
      imgSrc;

    elements.lightboxImage.style.transform =
      'scale(1)';

    elements.imageLightboxModal.classList.remove(
      'hidden'
    );
  }

  function closeLightbox() {
    elements.imageLightboxModal.classList.add(
      'hidden'
    );

    elements.lightboxImage.src =
      '';
  }

  function openModal(
    modalEl
  ) {
    modalEl.classList.remove(
      'hidden'
    );
  }

  function closeModal(
    modalEl
  ) {
    modalEl.classList.add(
      'hidden'
    );
  }

  // ==========================================================================
  // Toast
  // ==========================================================================

  function showToast(
    message,
    type = 'info'
  ) {
    const toast =
      document.createElement(
        'div'
      );

    toast.className =
      `toast ${type}`;

    toast.textContent =
      message;

    elements.toastContainer.appendChild(
      toast
    );

    setTimeout(
      () => {
        toast.style.opacity =
          '0';

        toast.style.transform =
          'translateX(100%)';

        setTimeout(
          () =>
            toast.remove(),
          250
        );
      },
      3200
    );
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================

  function setupEventListeners() {

    // ------------------------------------------------------------------------
    // Theme
    // ------------------------------------------------------------------------

    elements.themeToggleBtn.addEventListener(
      'click',
      toggleTheme
    );

    // ------------------------------------------------------------------------
    // Sidebar
    // ------------------------------------------------------------------------

    elements.sidebarToggleBtn.addEventListener(
      'click',
      () => {
        elements.sidebar.classList.toggle(
          'collapsed'
        );
      }
    );

    elements.sidebarCloseBtn.addEventListener(
      'click',
      () => {
        elements.sidebar.classList.add(
          'collapsed'
        );
      }
    );

    // ------------------------------------------------------------------------
    // New Chat
    // ------------------------------------------------------------------------

    elements.newChatBtn.addEventListener(
      'click',
      startNewChat
    );

    // ------------------------------------------------------------------------
    // Keyboard shortcut
    // ------------------------------------------------------------------------

    window.addEventListener(
      'keydown',
      e => {
        if (
          (e.metaKey ||
            e.ctrlKey) &&
          e.key === 'k'
        ) {
          e.preventDefault();

          startNewChat();
        }
      }
    );

    // ------------------------------------------------------------------------
    // Model selector
    // ------------------------------------------------------------------------

    elements.modelSelector.addEventListener(
      'change',
      handleModelChange
    );

    // ------------------------------------------------------------------------
    // Detection toggle
    // ------------------------------------------------------------------------

    if (
      elements.detectionToggle
    ) {
      elements.detectionToggle.addEventListener(
        'click',
        () => {
          state.useDetection =
            !state.useDetection;

          elements.detectionToggle.classList.toggle(
            'active',
            state.useDetection
          );

          showToast(
            `YOLO-OBB Object Detection ${
              state.useDetection
                ? 'enabled'
                : 'disabled'
            }`,
            'info'
          );
        }
      );
    }

    // ------------------------------------------------------------------------
    // RAG toggle
    // ------------------------------------------------------------------------

    if (
      elements.ragToggle
    ) {
      elements.ragToggle.addEventListener(
        'click',
        () => {
          state.useRag =
            !state.useRag;

          elements.ragToggle.classList.toggle(
            'active',
            state.useRag
          );

          handleModelChange();

          showToast(
            `Pinecone RAG ${
              state.useRag
                ? 'enabled'
                : 'disabled'
            }`,
            'info'
          );
        }
      );
    }

    // ------------------------------------------------------------------------
    // Web search
    // ------------------------------------------------------------------------

    elements.webSearchToggle.addEventListener(
      'click',
      () => {
        state.useWebSearch =
          !state.useWebSearch;

        elements.webSearchToggle.classList.toggle(
          'active',
          state.useWebSearch
        );

        showToast(
          `Web search ${
            state.useWebSearch
              ? 'enabled'
              : 'disabled'
          }`,
          'info'
        );
      }
    );

    // ------------------------------------------------------------------------
    // Clear chat
    // ------------------------------------------------------------------------

    elements.clearChatBtn.addEventListener(
      'click',
      () => {
        const chat =
          getCurrentChat();

        if (chat) {
          chat.messages =
            [];

          saveConversations();

          renderCurrentChat();

          showToast(
            'Chat cleared',
            'info'
          );
        }
      }
    );

    // ------------------------------------------------------------------------
    // Export chat
    // ------------------------------------------------------------------------

    elements.exportChatBtn.addEventListener(
      'click',
      () => {
        const chat =
          getCurrentChat();

        if (
          !chat ||
          chat.messages.length === 0
        ) {
          showToast(
            'No messages to export',
            'error'
          );

          return;
        }

        let mdContent =
          `# VisionOrbit Conversation: ${chat.title}\n` +
          `*Exported on ${new Date().toLocaleString()}*\n\n` +
          `---\n\n`;

        chat.messages.forEach(
          m => {
            mdContent +=
              `### ${
                m.role === 'user'
                  ? '👤 User'
                  : '🪐 VisionOrbit AI'
              }\n\n`;

            mdContent +=
              `${m.content}\n\n`;

            mdContent +=
              `---\n\n`;
          }
        );

        const blob =
          new Blob(
            [mdContent],
            {
              type:
                'text/markdown'
            }
          );

        const url =
          URL.createObjectURL(
            blob
          );

        const a =
          document.createElement(
            'a'
          );

        a.href =
          url;

        a.download =
          `VisionOrbit_${chat.title.replace(
            /[^a-zA-Z0-9]/g,
            '_'
          )}.md`;

        a.click();

        URL.revokeObjectURL(
          url
        );

        showToast(
          'Exported conversation as Markdown',
          'success'
        );
      }
    );

    // ------------------------------------------------------------------------
    // Prompt auto-resize
    // ------------------------------------------------------------------------

    elements.promptInput.addEventListener(
      'input',
      function () {
        this.style.height =
          'auto';

        this.style.height =
          Math.min(
            this.scrollHeight,
            180
          ) + 'px';
      }
    );

    // ------------------------------------------------------------------------
    // Enter to send
    // ------------------------------------------------------------------------

    elements.promptInput.addEventListener(
      'keydown',
      function (e) {
        if (
          e.key === 'Enter' &&
          !e.shiftKey
        ) {
          e.preventDefault();

          handleChatSubmit();
        }
      }
    );

    // ------------------------------------------------------------------------
    // Chat form
    // ------------------------------------------------------------------------

    elements.chatForm.addEventListener(
      'submit',
      handleChatSubmit
    );

    // ------------------------------------------------------------------------
    // Upload button
    // ------------------------------------------------------------------------

    elements.uploadBtn.addEventListener(
      'click',
      () => {
        elements.imageFileInput.click();
      }
    );

    // ------------------------------------------------------------------------
    // File input
    // ------------------------------------------------------------------------

    elements.imageFileInput.addEventListener(
      'change',
      e => {
        handleFileSelection(
          e.target.files
        );

        // Allow selecting the same
        // file again later.
        elements.imageFileInput.value =
          '';
      }
    );

    // ------------------------------------------------------------------------
    // Hero dropzone
    // ------------------------------------------------------------------------

    elements.heroDropzone.addEventListener(
      'click',
      () => {
        elements.imageFileInput.click();
      }
    );

    // ------------------------------------------------------------------------
    // Suggestion cards
    // ------------------------------------------------------------------------

    document
      .querySelectorAll(
        '.suggestion-card'
      )
      .forEach(
        card => {
          card.addEventListener(
            'click',
            () => {
              const promptText =
                card.getAttribute(
                  'data-prompt'
                );

              elements.promptInput.value =
                promptText;

              elements.promptInput.focus();

              elements.promptInput.dispatchEvent(
                new Event('input')
              );
            }
          );
        }
      );

    // ------------------------------------------------------------------------
    // Clipboard paste
    // ------------------------------------------------------------------------

    window.addEventListener(
      'paste',
      e => {
        const items =
          e.clipboardData
            ? e.clipboardData.items
            : [];

        for (
          let i = 0;
          i < items.length;
          i++
        ) {
          if (
            items[i].type.indexOf(
              'image'
            ) !== -1
          ) {
            const file =
              items[i].getAsFile();

            handleFileSelection(
              [file]
            );

            showToast(
              'Image pasted from clipboard',
              'info'
            );
          }
        }
      }
    );

    // ------------------------------------------------------------------------
    // Drag & Drop
    // ------------------------------------------------------------------------

    let dragCounter = 0;

    window.addEventListener(
      'dragenter',
      e => {
        e.preventDefault();

        dragCounter++;

        elements.dragDropOverlay.classList.add(
          'active'
        );
      }
    );

    window.addEventListener(
      'dragleave',
      e => {
        e.preventDefault();

        dragCounter--;

        if (
          dragCounter <= 0
        ) {
          dragCounter = 0;

          elements.dragDropOverlay.classList.remove(
            'active'
          );
        }
      }
    );

    window.addEventListener(
      'dragover',
      e => {
        e.preventDefault();
      }
    );

    window.addEventListener(
      'drop',
      e => {
        e.preventDefault();

        dragCounter = 0;

        elements.dragDropOverlay.classList.remove(
          'active'
        );

        if (
          e.dataTransfer &&
          e.dataTransfer.files.length > 0
        ) {
          handleFileSelection(
            e.dataTransfer.files
          );
        }
      }
    );

    // ------------------------------------------------------------------------
    // Voice recognition
    // ------------------------------------------------------------------------

    if (
      'webkitSpeechRecognition' in
        window ||
      'SpeechRecognition' in
        window
    ) {
      const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

      const recognition =
        new SpeechRecognition();

      recognition.continuous =
        false;

      recognition.interimResults =
        false;

      recognition.onresult =
        event => {
          const transcript =
            event.results[0][0]
              .transcript;

          elements.promptInput.value +=
            (
              elements.promptInput
                .value
                ? ' '
                : ''
            ) + transcript;

          elements.promptInput.dispatchEvent(
            new Event('input')
          );

          elements.voiceInputBtn.classList.remove(
            'active'
          );

          showToast(
            'Voice transcription captured',
            'info'
          );
        };

      recognition.onerror =
        () => {
          elements.voiceInputBtn.classList.remove(
            'active'
          );

          showToast(
            'Speech recognition error',
            'error'
          );
        };

      recognition.onend =
        () => {
          elements.voiceInputBtn.classList.remove(
            'active'
          );
        };

      elements.voiceInputBtn.addEventListener(
        'click',
        () => {
          if (
            elements.voiceInputBtn.classList.contains(
              'active'
            )
          ) {
            recognition.stop();

            elements.voiceInputBtn.classList.remove(
              'active'
            );

          } else {
            recognition.start();

            elements.voiceInputBtn.classList.add(
              'active'
            );

            showToast(
              'Listening... Speak now',
              'info'
            );
          }
        }
      );

    } else {
      elements.voiceInputBtn.title =
        'Speech recognition not supported in this browser';

      elements.voiceInputBtn.style.opacity =
        '0.5';
    }

    // ------------------------------------------------------------------------
    // Settings modal
    // ------------------------------------------------------------------------

    elements.openSettingsBtn.addEventListener(
      'click',
      () =>
        openModal(
          elements.settingsModal
        )
    );

    elements.closeSettingsModalBtn.addEventListener(
      'click',
      () =>
        closeModal(
          elements.settingsModal
        )
    );

    elements.settingsModal
      .querySelector(
        '.modal-backdrop'
      )
      .addEventListener(
        'click',
        () =>
          closeModal(
            elements.settingsModal
          )
      );

    elements.saveSettingsBtn.addEventListener(
      'click',
      saveSettings
    );

    elements.resetSettingsBtn.addEventListener(
      'click',
      resetSettings
    );

    // ------------------------------------------------------------------------
    // API key visibility
    // ------------------------------------------------------------------------

    elements.toggleKeyVisibility.addEventListener(
      'click',
      () => {
        const isPass =
          elements.openaiKeyInput
            .type ===
          'password';

        elements.openaiKeyInput.type =
          isPass
            ? 'text'
            : 'password';
      }
    );

    // ------------------------------------------------------------------------
    // Lightbox
    // ------------------------------------------------------------------------

    elements.lightboxCloseBtn.addEventListener(
      'click',
      closeLightbox
    );

    elements.imageLightboxModal
      .querySelector(
        '.lightbox-backdrop'
      )
      .addEventListener(
        'click',
        closeLightbox
      );

    elements.lightboxZoomIn.addEventListener(
      'click',
      () => {
        state.lightboxScale =
          Math.min(
            state.lightboxScale +
              0.3,
            3
          );

        elements.lightboxImage.style.transform =
          `scale(${state.lightboxScale})`;
      }
    );

    elements.lightboxZoomOut.addEventListener(
      'click',
      () => {
        state.lightboxScale =
          Math.max(
            state.lightboxScale -
              0.3,
            0.5
          );

        elements.lightboxImage.style.transform =
          `scale(${state.lightboxScale})`;
      }
    );

    // ------------------------------------------------------------------------
    // Scroll-to-bottom
    // ------------------------------------------------------------------------

    elements.chatContainer.addEventListener(
      'scroll',
      () => {
        const scrollPos =
          elements.chatContainer.scrollTop;

        const scrollHeight =
          elements.chatContainer.scrollHeight;

        const clientHeight =
          elements.chatContainer.clientHeight;

        if (
          scrollHeight -
            scrollPos -
            clientHeight >
          200
        ) {
          elements.scrollToBottomBtn.classList.remove(
            'hidden'
          );
        } else {
          elements.scrollToBottomBtn.classList.add(
            'hidden'
          );
        }
      }
    );

    elements.scrollToBottomBtn.addEventListener(
      'click',
      scrollChatToBottom
    );
  }

  // ==========================================================================
  // Start Application
  // ==========================================================================

  if (
    document.readyState ===
    'loading'
  ) {
    document.addEventListener(
      'DOMContentLoaded',
      initApp
    );
  } else {
    initApp();
  }

})();
