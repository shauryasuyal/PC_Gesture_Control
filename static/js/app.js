/**
 * FG Gesture Control — Core App Logic
 * Navigation, SocketIO, Toasts, Utilities
 */

// ── Socket.IO ────────────────────────────────
const socket = io();

// ── State ────────────────────────────────────
let currentPage = 'dashboard';
let engineStatus = {};

// ── DOM Ready ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initSocketIO();
    pollStatus();
});

// ═══════════════════════════════════════════════
// THEME SWITCHING
// ═══════════════════════════════════════════════
function initTheme() {
    const saved = localStorage.getItem('fg-theme') || 'dark';
    applyTheme(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('fg-theme', next);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const toggleBtn = document.getElementById('themeToggle');
    if (toggleBtn) toggleBtn.textContent = theme === 'dark' ? '🌙' : '☀️';
    const darkToggle = document.getElementById('toggleDarkMode');
    if (darkToggle) darkToggle.checked = theme === 'dark';
}

// ═══════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════
function initNavigation() {
    document.querySelectorAll('.topbar-link').forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.dataset.page;
            if (page === currentPage) return;
            switchPage(page);
        });
    });
}

function switchPage(page) {
    // Update nav buttons
    document.querySelectorAll('.topbar-link').forEach(b => b.classList.remove('active'));
    const activeBtn = document.querySelector(`.topbar-link[data-page="${page}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    // Switch pages with animation
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) {
        pageEl.classList.add('active');
        // Re-trigger animation
        pageEl.style.animation = 'none';
        pageEl.offsetHeight; // force reflow
        pageEl.style.animation = '';
    }

    currentPage = page;

    // Load page-specific data
    if (page === 'gestures') loadGestures();
    if (page === 'training') {
        loadSampleOverview();
        const btn = document.getElementById('trainButton');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🚀 Train Model';
        }
    }
}

// ═══════════════════════════════════════════════
// SOCKET.IO
// ═══════════════════════════════════════════════
function initSocketIO() {
    socket.on('connect', () => {
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
    });

    socket.on('status_update', (data) => {
        engineStatus = data;
        updateDashboardStatus(data);
    });

    socket.on('action_executed', (data) => {
        addActionLogEntry(data);
        showToast(`${data.gesture} → ${data.action}`, 'success');
    });
}

function updateConnectionStatus(online) {
    const el = document.getElementById('engineStatus');
    if (!el) return;
    const dot = el.querySelector('.status-dot');
    const txt = el.querySelector('.status-text');
    if (dot) dot.className = online ? 'status-dot online' : 'status-dot offline';
    if (txt) txt.textContent = online ? 'Connected' : 'Disconnected';
}

// ═══════════════════════════════════════════════
// STATUS POLLING
// ═══════════════════════════════════════════════
function pollStatus() {
    setInterval(async () => {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            engineStatus = data;
            updateDashboardStatus(data);
        } catch (e) { /* ignore */ }
    }, 2000);
}

// ═══════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════
async function switchMode(mode) {
    try {
        const res = await fetch('/api/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });
        const data = await res.json();
        if (data.success) {
            // Update mode switcher UI
            document.querySelectorAll('.mode-btn, .mode-pill').forEach(b => b.classList.remove('active'));
            const activeBtn = document.querySelector(`.mode-btn[data-mode="${mode}"], .mode-pill[data-mode="${mode}"]`);
            if (activeBtn) activeBtn.classList.add('active');
            showToast(`Switched to ${mode} mode`, 'info');
        }
    } catch (e) {
        showToast('Failed to switch mode', 'error');
    }
}

// ═══════════════════════════════════════════════
// RECOGNITION TOGGLE
// ═══════════════════════════════════════════════
async function toggleRecognition() {
    try {
        const res = await fetch('/api/engine/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();
        const state = data.custom_gesture_enabled ? 'enabled' : 'disabled';
        showToast(`Recognition ${state}`, 'info');
    } catch (e) {
        showToast('Failed to toggle recognition', 'error');
    }
}

// ═══════════════════════════════════════════════
// MODALS
// ═══════════════════════════════════════════════
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.remove('show');
        document.body.style.overflow = '';
    }
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay') && e.target.classList.contains('show')) {
        e.target.classList.remove('show');
        document.body.style.overflow = '';
    }
});

// ═══════════════════════════════════════════════
// TOASTS
// ═══════════════════════════════════════════════
function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toastContainer');
    const icons = {
        success: '✅',
        error: '❌',
        info: 'ℹ️',
        warning: '⚠️'
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ═══════════════════════════════════════════════
// CONFIRM DIALOG
// ═══════════════════════════════════════════════
function showConfirm(title, message, onConfirm) {
    document.getElementById('confirmTitle').textContent = title;
    document.getElementById('confirmMessage').textContent = message;

    const confirmBtn = document.getElementById('confirmAction');
    // Remove old listener by replacing node
    const newBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newBtn, confirmBtn);
    newBtn.addEventListener('click', () => {
        onConfirm();
        closeModal('confirmDialog');
    });

    openModal('confirmDialog');
}