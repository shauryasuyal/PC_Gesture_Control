/**
 * FG Gesture Control — Gestures Page Logic
 * Built-in + custom gesture cards, add/record/delete
 */

// ── State ────────────────────────────────────
let gestureConfig = null;
let selectedIcon = '✋';
let selectedActionType = 'preset';
let gestureShapes = {};

// ═══════════════════════════════════════════════
// LOAD GESTURES
// ═══════════════════════════════════════════════
async function loadGestures() {
    try {
        const [gestRes, shapesRes] = await Promise.all([
            fetch('/api/gestures'),
            fetch('/api/gesture_shapes')
        ]);
        gestureConfig = await gestRes.json();
        gestureShapes = await shapesRes.json();
        renderBuiltinGestures();
        renderCustomGestures();
    } catch (e) {
        showToast('Failed to load gestures', 'error');
    }
}

// ═══════════════════════════════════════════════
// RENDER BUILT-IN GESTURES
// ═══════════════════════════════════════════════
function renderBuiltinGestures() {
    const grid = document.getElementById('builtinGestureGrid');
    if (!grid || !gestureConfig) return;

    // Find the current active mode
    const activeMode = document.querySelector('.mode-btn.active');
    const modeKey = activeMode ? activeMode.dataset.mode : 'CURSOR';

    const modeMap = {
        'CURSOR': 'Cursor Mode (C)',
        'DRAWING': 'Drawing Mode (X)'
    };

    const modeName = modeMap[modeKey] || 'Cursor Mode (C)';
    const modeData = (gestureConfig.builtin_modes || []).find(m => m.name === modeName);

    if (!modeData || !modeData.gestures.length) {
        grid.innerHTML = '<div class="gesture-grid-empty">No built-in gestures for this mode.</div>';
        return;
    }

    const modeIcons = {
        'Move cursor': '🖐️', 'Left click': '🤏', 'Double left click': '🖱️🖱️', 'Right click': '🖱️',
        'Scroll up': '⬆️', 'Scroll down': '⬇️', 'Task View (Win+Tab)': '🖥️',
        'Volume control': '🔊', 'Voice dictation (Win+H)': '🎤',
        'Play notes C4–C5': '🎵', 'Draw': '✏️', 'Adjust brush size': '🎨',
        'Select color': '🌈', 'Eraser': '🧹', 'Clear canvas': '🗑️',
        'Show all 21 landmarks + data': '🔬'
    };

    grid.innerHTML = modeData.gestures.map(g => `
        <div class="gesture-card">
            <div class="gesture-type-badge">Built-in</div>
            <div class="gesture-card-icon">${modeIcons[g.action] || '✋'}</div>
            <div class="gesture-card-name">${g.name}</div>
            <div class="gesture-card-action">${g.action}</div>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════
// RENDER CUSTOM GESTURES
// ═══════════════════════════════════════════════
function renderCustomGestures() {
    const grid = document.getElementById('customGestureGrid');
    if (!grid || !gestureConfig) return;

    const gestures = gestureConfig.gestures || [];

    if (!gestures.length) {
        grid.innerHTML = '<div class="gesture-grid-empty">No custom gestures yet. Click "Add Custom Gesture" to create one.</div>';
        return;
    }

    grid.innerHTML = gestures.map(g => {
        const actionLabel = g.action_label || g.action_value || 'Not configured';
        const typeLabel = { shortcut: '⌨️', app: '📂', command: '💻', preset: '⚡' }[g.action_type] || '⚡';

        return `
            <div class="gesture-card">
                <div class="gesture-type-badge">${typeLabel} ${g.action_type}</div>
                <div class="gesture-card-icon">${g.icon || '👋'}</div>
                <div class="gesture-card-name">${g.name}</div>
                <div class="gesture-card-action">${actionLabel}</div>
                <div class="gesture-card-samples">${g.samples_count || 0} samples recorded</div>
                <div class="gesture-card-actions">
                    <button class="btn btn-primary btn-sm" onclick="startRecording('${g.id}', '${g.name}')">🎯 Record</button>
                    <button class="btn btn-ghost btn-sm" onclick="deleteGesture('${g.id}', '${g.name}')">🗑️</button>
                </div>
            </div>
        `;
    }).join('');
}

// ═══════════════════════════════════════════════
// MODE SWITCH (update built-in grid)
// ═══════════════════════════════════════════════
// Override the global switchMode to also re-render built-in gestures
const _origSwitchMode = window.switchMode;
window.switchMode = async function (mode) {
    await _origSwitchMode(mode);
    renderBuiltinGestures();
};

// ═══════════════════════════════════════════════
// ADD GESTURE MODAL
// ═══════════════════════════════════════════════
async function showAddGestureModal() {
    // Reset form
    document.getElementById('gestureName').value = '';
    document.getElementById('gestureShortcut') && (document.getElementById('gestureShortcut').value = '');
    document.getElementById('gestureApp') && (document.getElementById('gestureApp').value = '');
    document.getElementById('gestureCommand') && (document.getElementById('gestureCommand').value = '');

    // Build the shape dropdown with two sections:
    //  1. Named emoji shapes (excluding those used by built-ins)
    //  2. 10 free-form "Custom Gesture" slots with no shape constraint
    const builtinShapes = [
        'fist','pointing','open_palm','peace_sign','three_fingers',
        'four_fingers','index_pointing_up','thumbs_up'
    ];

    const shapeSelect = document.getElementById('gestureShape');
    shapeSelect.innerHTML = '<option value="">-- Choose a gesture type --</option>';

    // ── Section 1: Named emoji shapes ──
    const namedGroup = document.createElement('optgroup');
    namedGroup.label = '🤚 Named Hand Shapes';
    for (const [id, shape] of Object.entries(gestureShapes)) {
        if (builtinShapes.includes(id)) continue;
        if (shape.freeform) continue;  // skip freeform — goes in section 2
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${shape.icon} ${shape.name}`;
        namedGroup.appendChild(opt);
    }
    shapeSelect.appendChild(namedGroup);

    // ── Section 2: Free-form custom slots ──
    const freeGroup = document.createElement('optgroup');
    freeGroup.label = '✨ Custom (any gesture you invent)';
    for (const [id, shape] of Object.entries(gestureShapes)) {
        if (!shape.freeform) continue;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = `${shape.icon} ${shape.name}`;
        freeGroup.appendChild(opt);
    }
    shapeSelect.appendChild(freeGroup);

    shapeSelect.onchange = () => {
        const desc = document.getElementById('gestureShapeDesc');
        const shape = gestureShapes[shapeSelect.value];
        if (!shape) { desc.textContent = ''; return; }
        desc.textContent = shape.desc;
        // Show/hide freeform warning
        const warn = document.getElementById('gestureShapeFreeformWarning');
        if (warn) warn.style.display = shape.freeform ? 'block' : 'none';
    };

    // Populate presets
    const presetSelect = document.getElementById('gesturePreset');
    presetSelect.innerHTML = '';
    const presets = gestureConfig?.action_presets || {};
    for (const [id, preset] of Object.entries(presets)) {
        presetSelect.innerHTML += `<option value="${id}">${preset.label}</option>`;
    }

    // Reset action type
    setActionType('preset');

    openModal('addGestureModal');
}

// ── Action Type Tabs ─────────────────────────
function setActionType(type) {
    selectedActionType = type;
    document.querySelectorAll('.action-tab, .action-type-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.type === type);
    });
    document.getElementById('presetGroup').style.display = type === 'preset' ? '' : 'none';
    document.getElementById('shortcutGroup').style.display = type === 'shortcut' ? '' : 'none';
    document.getElementById('appGroup').style.display = type === 'app' ? '' : 'none';
    document.getElementById('commandGroup').style.display = type === 'command' ? '' : 'none';
}



// ═══════════════════════════════════════════════
// ADD GESTURE (API)
// ═══════════════════════════════════════════════
async function addGesture() {
    const name = document.getElementById('gestureName').value.trim();
    if (!name) {
        showToast('Please enter a gesture name', 'warning');
        return;
    }

    const shape = document.getElementById('gestureShape').value;

    let actionType, actionValue, actionLabel;

    if (selectedActionType === 'preset') {
        const presetSelect = document.getElementById('gesturePreset');
        const presetId = presetSelect.value;
        const preset = gestureConfig?.action_presets?.[presetId];
        actionType = preset?.type || 'shortcut';
        actionValue = preset?.value || '';
        actionLabel = preset?.label || '';
    } else if (selectedActionType === 'shortcut') {
        actionType = 'shortcut';
        actionValue = document.getElementById('gestureShortcut').value.trim();
        actionLabel = `Shortcut: ${actionValue}`;
    } else if (selectedActionType === 'app') {
        actionType = 'app';
        actionValue = document.getElementById('gestureApp').value.trim();
        actionLabel = `Launch: ${actionValue}`;
    } else {
        actionType = 'command';
        actionValue = document.getElementById('gestureCommand').value.trim();
        actionLabel = `Run: ${actionValue}`;
    }

    try {
        const res = await fetch('/api/gestures', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                gesture_shape: shape,
                action_type: actionType,
                action_value: actionValue,
                action_label: actionLabel
            })
        });

        if (res.ok) {
            const created = await res.json();
            closeModal('addGestureModal');
            showToast(`Gesture "${name}" created! Opening recorder...`, 'success');
            await loadGestures();
            // Auto-open recording modal for the newly created gesture
            startRecording(created.id, created.name);
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to create gesture', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    }
}

// ═══════════════════════════════════════════════
// DELETE GESTURE
// ═══════════════════════════════════════════════
function deleteGesture(id, name) {
    showConfirm('Delete Gesture', `Are you sure you want to delete "${name}"? This will also remove all recorded samples.`, async () => {
        try {
            const res = await fetch(`/api/gestures/${id}`, { method: 'DELETE' });
            if (res.ok) {
                showToast(`Gesture "${name}" deleted`, 'info');
                await loadGestures();
            } else {
                showToast('Failed to delete gesture', 'error');
            }
        } catch (e) {
            showToast('Network error', 'error');
        }
    });
}

// ═══════════════════════════════════════════════
// RECORDING
// ═══════════════════════════════════════════════
let recordingGestureId = null;

async function startRecording(gestureId, gestureName) {
    recordingGestureId = gestureId;
    document.getElementById('recordGestureName').textContent = gestureName;
    document.getElementById('recordCount').textContent = '0';
    document.getElementById('recordProgressFill').style.width = '0%';

    try {
        const res = await fetch(`/api/gestures/${gestureId}/record`, { method: 'POST' });
        if (res.ok) {
            openModal('recordModal');

            // Remove any stale listener before adding a fresh one
            socket.off('recording_progress');

            // Listen for progress
            socket.on('recording_progress', (data) => {
                if (data.gesture_id === recordingGestureId) {
                    document.getElementById('recordCount').textContent = data.count;
                    const progress = Math.min((data.count / 400) * 100, 100);
                    document.getElementById('recordProgressFill').style.width = `${progress}%`;
                }
            });
        } else {
            showToast('Failed to start recording', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    }
}

async function stopRecordingAndClose() {
    if (recordingGestureId) {
        try {
            await fetch(`/api/gestures/${recordingGestureId}/stop`, { method: 'POST' });
            showToast('Samples saved!', 'success');
        } catch (e) {
            showToast('Failed to stop recording', 'error');
        }
    }

    socket.off('recording_progress');
    recordingGestureId = null;
    closeModal('recordModal');
    await loadGestures();
    // After recording, take user to Training page to run the model.
    switchPage('training');
}

// ── Initial load when page is first shown ────
document.addEventListener('DOMContentLoaded', () => {
    if (currentPage === 'gestures') loadGestures();
});