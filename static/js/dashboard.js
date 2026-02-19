/**
 * FG Gesture Control — Dashboard Page Logic
 * Real-time status, camera feed, action log
 */

// ═══════════════════════════════════════════════
// POWER TOGGLE
// ═══════════════════════════════════════════════
function togglePower() {
    fetch('/api/engine/power', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            const btn = document.getElementById('powerToggleBtn');
            const label = document.getElementById('powerLabel');
            if (data.paused) {
                btn.classList.remove('active');
                btn.classList.add('off');
                label.textContent = 'System Paused';
                label.style.color = 'var(--danger)';
                showToast('Camera paused', 'warning');
            } else {
                btn.classList.add('active');
                btn.classList.remove('off');
                label.textContent = 'System Active';
                label.style.color = 'var(--success)';
                showToast('Camera resumed', 'success');
            }
        })
        .catch(() => showToast('Failed to toggle power', 'error'));
}

// ═══════════════════════════════════════════════
// STATUS UPDATES
// ═══════════════════════════════════════════════
function updateDashboardStatus(data) {
    // Power state
    const powerBtn = document.getElementById('powerToggleBtn');
    const powerLabel = document.getElementById('powerLabel');
    if (powerBtn) {
        if (data.paused) {
            powerBtn.classList.remove('active');
            powerBtn.classList.add('off');
            if (powerLabel) {
                powerLabel.textContent = 'System Paused';
                powerLabel.style.color = 'var(--danger)';
            }
        } else {
            powerBtn.classList.add('active');
            powerBtn.classList.remove('off');
            if (powerLabel) {
                powerLabel.textContent = 'System Active';
                powerLabel.style.color = 'var(--success)';
            }
        }
    }

    // Current mode
    const modeNames = {
        'CURSOR': 'Cursor',
        'DRAWING': 'Drawing'
    };
    const modeEl = document.getElementById('currentMode');
    if (modeEl) modeEl.textContent = modeNames[data.current_mode] || data.current_mode;

    // Current gesture
    const gestureEl = document.getElementById('currentGesture');
    if (gestureEl) {
        gestureEl.textContent = data.current_custom_gesture || 'No gesture detected';
        gestureEl.classList.toggle('detected', !!data.current_custom_gesture);
    }

    // Confidence
    const confFill = document.getElementById('confidenceFill');
    if (confFill) {
        const conf = data.current_custom_confidence || 0;
        confFill.style.width = `${conf}%`;
    }

    // FPS
    const fpsEl = document.getElementById('fpsValue');
    if (fpsEl) fpsEl.textContent = data.paused ? '—' : (data.fps || 0);

    // Model status
    const modelEl = document.getElementById('modelStatus');
    if (modelEl) {
        if (data.model_loaded) {
            modelEl.textContent = '✓';
            modelEl.style.color = 'var(--success)';
        } else {
            modelEl.textContent = '✗';
            modelEl.style.color = 'var(--danger)';
        }
    }

    // Camera overlay
    const overlay = document.getElementById('cameraOverlay');
    if (overlay) {
        if (data.paused) {
            overlay.classList.remove('hidden');
            overlay.querySelector('span').textContent = '⏸ Camera paused';
        } else if (data.running && data.hand_detected) {
            overlay.classList.add('hidden');
        } else if (data.running) {
            overlay.classList.remove('hidden');
            overlay.querySelector('span').textContent = 'Show your hand...';
        } else {
            overlay.classList.remove('hidden');
            overlay.querySelector('span').textContent = 'Starting camera...';
        }
    }

    // Update mode switcher buttons to match current mode
    document.querySelectorAll('.mode-btn, .mode-pill').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === data.current_mode);
    });
}

// ═══════════════════════════════════════════════
// ACTION LOG
// ═══════════════════════════════════════════════
function addActionLogEntry(entry) {
    const log = document.getElementById('actionLog');
    if (!log) return;

    // Remove empty state
    const empty = log.querySelector('.log-empty');
    if (empty) empty.remove();

    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML = `
        <span class="log-gesture">${entry.gesture}</span>
        <span class="log-action">${entry.action}</span>
        <span class="log-confidence">${entry.confidence}%</span>
        <span class="log-time">${entry.time}</span>
    `;

    // Prepend (newest first)
    log.insertBefore(el, log.firstChild);

    // Limit to 20
    while (log.children.length > 20) {
        log.removeChild(log.lastChild);
    }
}

function clearActionLog() {
    const log = document.getElementById('actionLog');
    if (log) {
        log.innerHTML = '<div class="log-empty">No actions yet. Use gestures to trigger desktop actions.</div>';
    }
}