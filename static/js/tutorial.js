/**
 * FG Gesture Control — Tutorial Page Logic
 * Animated cursor in Windows mockup, live taskbar clock,
 * interactive built-in gesture practice (click → learn → try → use).
 */

// ── Animated Cursor in Windows Mockup ────────
let cursorAnimation = null;

function startCursorAnimation() {
    const cursor = document.getElementById('winCursor');
    if (!cursor) return;

    const positions = [
        { top: '60px', left: '200px' },   // Near window
        { top: '120px', left: '350px' },   // Inside window content
        { top: '90px', left: '280px' },    // Title bar area
        { top: '180px', left: '400px' },   // Mid window
        { top: '40px', left: '50px' },     // Desktop icon
        { top: '150px', left: '300px' },   // Back to window
    ];

    let idx = 0;
    function moveCursor() {
        const pos = positions[idx % positions.length];
        cursor.style.top = pos.top;
        cursor.style.left = pos.left;
        idx++;
    }

    moveCursor();
    cursorAnimation = setInterval(moveCursor, 2500);
}

function stopCursorAnimation() {
    if (cursorAnimation) {
        clearInterval(cursorAnimation);
        cursorAnimation = null;
    }
}

// ── Taskbar Clock ────────────────────────────
let clockInterval = null;

function startTaskbarClock() {
    const timeEl = document.getElementById('taskbarTime');
    if (!timeEl) return;

    function updateClock() {
        const now = new Date();
        const time = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
        const date = now.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' });
        timeEl.innerHTML = `${time}<br>${date}`;
    }

    updateClock();
    clockInterval = setInterval(updateClock, 30000);
}

function stopTaskbarClock() {
    if (clockInterval) {
        clearInterval(clockInterval);
        clockInterval = null;
    }
}

// ── Built-in Gesture Practice ─────────────────
const LEARNED_KEY = 'fg-learned-builtin-gestures-v1';
const HOLD_TO_COMPLETE_MS = 2000;
const PINCH_THRESH = 0.06; // normalized distance (0..1) between thumb+index tips
const THUMB_PINKY_THRESH = 0.07; // normalized distance between thumb+pinky tips
const PALM_SPAN_THRESH = 0.22; // normalized x-span between index_mcp and pinky_mcp

let practice = {
    active: false,
    gesture: null,
    requiredFingers: null,
    mode: 'CURSOR',
    steps: [],
    action: '',
    lastOkAt: 0,
    okStartAt: 0,
    holdMs: 0,
    lastFingerState: null,
    lastMetrics: null,
    metricsBaseline: null,
    onFingerState: null,
    onMetrics: null,
};

function loadLearnedSet() {
    try {
        const raw = localStorage.getItem(LEARNED_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return new Set(Array.isArray(arr) ? arr : []);
    } catch {
        return new Set();
    }
}

function saveLearnedSet(set) {
    localStorage.setItem(LEARNED_KEY, JSON.stringify(Array.from(set)));
}

function markLearned(gestureId) {
    const learned = loadLearnedSet();
    learned.add(gestureId);
    saveLearnedSet(learned);
    refreshTutorialCardBadges();
}

function isLearned(gestureId) {
    return loadLearnedSet().has(gestureId);
}

function refreshTutorialCardBadges() {
    const learned = loadLearnedSet();
    document.querySelectorAll('.tutorial-gesture-card').forEach(card => {
        const id = card.dataset.gesture;
        card.classList.toggle('learned', learned.has(id));
        const badge = card.querySelector('.tutorial-learned-badge');
        if (badge) badge.remove();
        if (learned.has(id)) {
            const b = document.createElement('span');
            b.className = 'tutorial-learned-badge';
            b.textContent = '✓ Learned';
            card.appendChild(b);
        }
    });
}

function renderTutorialGestures() {
    const cursorWrap = document.getElementById('tutorialCursorGestures');
    const drawingWrap = document.getElementById('tutorialDrawingGestures');
    if (!cursorWrap || !drawingWrap) return;

    const byMode = {
        CURSOR: [],
        DRAWING: [],
    };

    // Note: requiredFingers values are finger UP states (true = extended).
    // Use null for "don't care".
    const gestures = [
        {
            id: 'move_cursor',
            mode: 'CURSOR',
            icon: '🖐️',
            name: 'Open Palm (Cursor)',
            desc: 'Simply open your palm naturally to move the cursor.',
            action: 'Move Cursor',
            requiredFingers: { thumb: true, index: true, middle: true, ring: true, pinky: true },
            steps: [
                'Relax your hand and open your palm toward the camera.',
                'Keep all fingers comfortably extended (no fist or partial close).',
                'Move your hand smoothly to move the cursor.',
            ],
        },
        {
            id: 'pinch_click',
            mode: 'CURSOR',
            icon: '🤏',
            name: 'Pinch (Thumb + Index)',
            desc: 'Pinch thumb + index fingertips together to left-click.',
            action: 'Left Click',
            requiredFingers: { thumb: null, index: null, middle: null, ring: null, pinky: null },
            steps: [
                'Open your hand naturally.',
                'Bring your thumb and index fingertips close together.',
                'Pinch until they nearly touch, then release.',
            ],
            validator: (finger, metrics) => (metrics?.pinch_thumb_index ?? 1) < PINCH_THRESH,
        },
        {
            id: 'double_click',
            mode: 'CURSOR',
            icon: '🖐️⬇️',
            name: 'Middle Finger Down',
            desc: 'Curl ONLY your middle finger down while keeping the others up.',
            action: 'Double Click',
            requiredFingers: { thumb: null, index: true, middle: false, ring: true, pinky: true },
            steps: [
                'Open your hand with fingers extended.',
                'Curl ONLY your middle finger down.',
                'Hold steady for a moment to trigger a double-click.',
            ],
        },
        {
            id: 'right_click',
            mode: 'CURSOR',
            icon: '💍⬇️',
            name: 'Ring Finger Down',
            desc: 'Curl ONLY your ring finger down while keeping the others up.',
            action: 'Right Click',
            requiredFingers: { thumb: null, index: true, middle: true, ring: false, pinky: true },
            steps: [
                'Open your hand with fingers extended.',
                'Curl ONLY your ring finger down.',
                'Hold steady briefly to right-click.',
            ],
        },
        {
            id: 'scroll_up',
            mode: 'CURSOR',
            icon: '⬆️',
            name: 'Thumb Cross (Scroll Up)',
            desc: 'Close your thumb and keep the rest of your fingers open.',
            action: 'Scroll Up',
            requiredFingers: { thumb: null, index: null, middle: true, ring: null, pinky: true },
            steps: [
                'Keep your hand open (avoid fist / transitions).',
                'Close your thumb.',
                'Hold the position to keep scrolling up.',
            ],
            validator: (finger, metrics) => {
                const cross = metrics?.thumb_cross_index;
                const okFingers = (finger?.middle === true) && (finger?.pinky === true);
                return !!cross && okFingers;
            }
        },
        {
            id: 'scroll_down',
            mode: 'CURSOR',
            icon: '⬇️',
            name: 'Pinky Down',
            desc: 'Curl ONLY your pinky down to scroll down continuously.',
            action: 'Scroll Down',
            requiredFingers: { thumb: null, index: null, middle: null, ring: null, pinky: false },
            steps: [
                'Open hand with all fingers up.',
                'Curl ONLY your pinky finger down.',
                'Hold to scroll down continuously.',
            ],
        },
        {
            id: 'task_view',
            mode: 'CURSOR',
            icon: '🖥️',
            name: 'Thumb + Pinky Tips Together',
            desc: 'Touch thumb + pinky tips together (hand spread) to open Task View.',
            action: 'Task View',
            requiredFingers: { thumb: null, index: null, middle: null, ring: null, pinky: null },
            steps: [
                'Spread your hand naturally.',
                'Bring your thumb tip and pinky tip close together.',
                'Hold briefly to open Task View.',
            ],
            validator: (finger, metrics) => {
                const tp = metrics?.pinch_thumb_pinky ?? 1;
                const span = metrics?.palm_span ?? 0;
                return tp < THUMB_PINKY_THRESH && span > PALM_SPAN_THRESH;
            }
        },
        {
            id: 'voice_dictation',
            mode: 'CURSOR',
            icon: '🎤',
            name: 'Index Only (Dictation)',
            desc: 'Index up, all other fingers closed to start Windows dictation.',
            action: 'Voice Dictation',
            requiredFingers: { thumb: false, index: true, middle: false, ring: false, pinky: false },
            steps: [
                'Close your hand into a fist.',
                'Extend ONLY your index finger straight up.',
                'Hold steady for a moment to trigger dictation.',
            ],
        },
        {
            id: 'volume',
            mode: 'CURSOR',
            icon: '✊🔄',
            name: 'Fist + Rotate',
            desc: 'Make a fist and rotate your wrist left/right to change volume.',
            action: 'Volume Control',
            requiredFingers: { thumb: null, index: false, middle: false, ring: false, pinky: false },
            steps: [
                'Close your hand into a fist.',
                'Rotate your wrist left/right.',
                'Clockwise increases volume; counter-clockwise decreases.',
            ],
        },

        // Drawing mode (AirDraw-style)
        {
            id: 'draw',
            mode: 'DRAWING',
            icon: '✏️',
            name: '1 Finger (Draw)',
            desc: 'Extend ONLY your index finger to draw.',
            action: 'Draw',
            requiredFingers: { thumb: null, index: true, middle: false, ring: false, pinky: false },
            steps: [
                'Switch to Drawing mode first.',
                'Extend ONLY your index finger (others relaxed).',
                'Move your index fingertip to draw on the canvas.',
            ],
        },
        {
            id: 'color_red',
            mode: 'DRAWING',
            icon: '🟥',
            name: '2 Fingers (Red)',
            desc: 'Extend index + middle fingers to switch to RED.',
            action: 'Set Color: Red',
            requiredFingers: { thumb: null, index: true, middle: true, ring: false, pinky: false },
            steps: [
                'Switch to Drawing mode first.',
                'Extend your index and middle fingers.',
                'Hold briefly to set brush colour to RED.',
            ],
        },
        {
            id: 'color_blue',
            mode: 'DRAWING',
            icon: '🟦',
            name: '3 Fingers (Blue)',
            desc: 'Extend index + middle + ring fingers to switch to BLUE.',
            action: 'Set Color: Blue',
            requiredFingers: { thumb: null, index: true, middle: true, ring: true, pinky: false },
            steps: [
                'Switch to Drawing mode first.',
                'Extend index, middle, and ring fingers.',
                'Hold briefly to set brush colour to BLUE.',
            ],
        },
        {
            id: 'color_green',
            mode: 'DRAWING',
            icon: '🟩',
            name: '4 Fingers (Green)',
            desc: 'Extend index + middle + ring + pinky fingers to switch to GREEN.',
            action: 'Set Color: Green',
            requiredFingers: { thumb: null, index: true, middle: true, ring: true, pinky: true },
            steps: [
                'Switch to Drawing mode first.',
                'Extend index, middle, ring, and pinky fingers.',
                'Hold briefly to set brush colour to GREEN.',
            ],
        },
        {
            id: 'erase',
            mode: 'DRAWING',
            icon: '🧽',
            name: 'Open Palm (Erase)',
            desc: 'Open your whole palm to erase parts of the drawing.',
            action: 'Erase',
            requiredFingers: { thumb: true, index: true, middle: true, ring: true, pinky: true },
            steps: [
                'Switch to Drawing mode first.',
                'Open your hand (all five fingers up).',
                'Move your palm over lines to erase them.',
            ],
        },
        {
            id: 'clear_canvas',
            mode: 'DRAWING',
            icon: '✊',
            name: 'Fist (Clear All)',
            desc: 'Make a fist to clear the whole canvas.',
            action: 'Clear Canvas',
            requiredFingers: { thumb: null, index: false, middle: false, ring: false, pinky: false },
            steps: [
                'Switch to Drawing mode first.',
                'Close your hand into a fist.',
                'The whole canvas is cleared.',
            ],
        },
    ];

    for (const g of gestures) byMode[g.mode].push(g);

    function cardHtml(g) {
        const learned = isLearned(g.id);
        return `
            <div class="tutorial-gesture-card ${learned ? 'learned' : ''}" data-gesture="${g.id}" data-mode="${g.mode}">
                <div class="tutorial-hand">${g.icon}</div>
                <div class="tutorial-info">
                    <div class="tutorial-gesture-name">${g.name}</div>
                    <div class="tutorial-gesture-desc">${g.desc}</div>
                </div>
                <span class="tutorial-action-badge">${g.action}</span>
                <span class="try-badge">${learned ? 'Practice again →' : 'Try it →'}</span>
                ${learned ? '<span class="tutorial-learned-badge">✓ Learned</span>' : ''}
            </div>
        `;
    }

    cursorWrap.innerHTML = byMode.CURSOR.map(cardHtml).join('');
    drawingWrap.innerHTML = byMode.DRAWING.map(cardHtml).join('');

    // Attach click handlers
    document.querySelectorAll('.tutorial-gesture-card').forEach(card => {
        card.addEventListener('click', () => {
            const id = card.dataset.gesture;
            const gesture = gestures.find(x => x.id === id);
            if (gesture) openPracticeModal(gesture);
        });
    });
}

function openPracticeModal(gesture) {
    practice.active = false;
    practice.gesture = gesture;
    practice.requiredFingers = gesture.requiredFingers || {};
    practice.mode = gesture.mode;
    practice.steps = gesture.steps || [];
    practice.action = gesture.action || '';
    practice.lastOkAt = 0;
    practice.okStartAt = 0;
    practice.holdMs = 0;
    practice.lastFingerState = null;
    practice.lastMetrics = null;
    practice.metricsBaseline = null;

    document.getElementById('practiceTitle').textContent = `Practice: ${gesture.name}`;
    document.getElementById('practiceIcon').textContent = gesture.icon || '✋';
    document.getElementById('practiceName').textContent = gesture.name || 'Gesture';
    document.getElementById('practiceAction').textContent = gesture.action || '';

    // Steps
    const stepsWrap = document.getElementById('practiceSteps');
    stepsWrap.innerHTML = practice.steps.map((s, i) => `
        <div class="practice-step">
            <div class="practice-step-num">${i + 1}</div>
            <div class="practice-step-text">${s}</div>
        </div>
    `).join('');

    // Required fingers
    const dots = {
        thumb: document.getElementById('fingerThumb'),
        index: document.getElementById('fingerIndex'),
        middle: document.getElementById('fingerMiddle'),
        ring: document.getElementById('fingerRing'),
        pinky: document.getElementById('fingerPinky'),
    };
    for (const [k, el] of Object.entries(dots)) {
        if (!el) continue;
        el.classList.remove('on', 'off');
        const expected = practice.requiredFingers?.[k];
        if (expected === true) el.classList.add('on');
        else if (expected === false) el.classList.add('off');
    }

    // Reset camera section
    const camSection = document.getElementById('practiceCameraSection');
    camSection.style.display = 'none';
    document.getElementById('practiceProgressFill').style.width = '0%';
    document.getElementById('practiceStatus').textContent = `Hold the gesture for ${Math.round(HOLD_TO_COMPLETE_MS / 1000)} seconds...`;
    document.getElementById('feedbackIcon').textContent = '🎯';
    document.getElementById('feedbackText').textContent = 'Click “Try it with Camera” to begin.';
    const useBtn = document.getElementById('practiceUseBtn');
    useBtn.style.display = 'none';

    openModal('practiceModal');
}

function closePracticeModal() {
    stopPractice();
    closeModal('practiceModal');
}

function startPractice() {
    if (!practice.gesture) return;

    const camSection = document.getElementById('practiceCameraSection');
    camSection.style.display = '';
    document.getElementById('feedbackIcon').textContent = '📸';
    document.getElementById('feedbackText').textContent = 'Show your hand to the camera...';
    document.getElementById('practiceProgressFill').style.width = '0%';
    document.getElementById('practiceUseBtn').style.display = 'none';
    practice.active = true;
    practice.okStartAt = 0;
    practice.holdMs = 0;
    practice.metricsBaseline = null;

    // Detach previous handlers if any
    if (practice.onFingerState) socket.off('finger_state', practice.onFingerState);
    if (practice.onMetrics) socket.off('tutorial_metrics', practice.onMetrics);

    practice.onFingerState = (data) => {
        practice.lastFingerState = data;
        updateLiveFingerState(data);
        tickPractice();
    };
    practice.onMetrics = (data) => {
        practice.lastMetrics = data;
        if (!practice.metricsBaseline && data) {
            practice.metricsBaseline = { ...data };
        }
        tickPractice();
    };

    socket.on('finger_state', practice.onFingerState);
    socket.on('tutorial_metrics', practice.onMetrics);
}

function stopPractice() {
    practice.active = false;
    if (practice.onFingerState) socket.off('finger_state', practice.onFingerState);
    if (practice.onMetrics) socket.off('tutorial_metrics', practice.onMetrics);
    practice.onFingerState = null;
    practice.onMetrics = null;
}

function updateLiveFingerState(state) {
    const map = {
        thumb: document.getElementById('liveThumb'),
        index: document.getElementById('liveIndex'),
        middle: document.getElementById('liveMiddle'),
        ring: document.getElementById('liveRing'),
        pinky: document.getElementById('livePinky'),
    };
    for (const [k, el] of Object.entries(map)) {
        if (!el) continue;
        const isUp = !!state?.[k];
        el.classList.toggle('on', isUp);
        el.classList.toggle('off', !isUp);
    }
}

function fingersMatch(required, actual) {
    if (!required) return true;
    for (const [k, expected] of Object.entries(required)) {
        if (expected === null || typeof expected === 'undefined') continue;
        if (!!actual?.[k] !== expected) return false;
    }
    return true;
}

function gestureSatisfied(gesture, finger, metrics, baseline) {
    if (!gesture) return false;

    // If gesture provides a validator, use it (still requires required finger constraints, if any).
    const fingerOk = fingersMatch(gesture.requiredFingers, finger);
    if (!fingerOk) return false;
    if (typeof gesture.validator === 'function') {
        return !!gesture.validator(finger, metrics, baseline);
    }
    return true;
}

function tickPractice() {
    if (!practice.active || !practice.gesture) return;

    const now = Date.now();
    const ok = gestureSatisfied(practice.gesture, practice.lastFingerState, practice.lastMetrics, practice.metricsBaseline);

    if (ok) {
        if (!practice.okStartAt) practice.okStartAt = now;
        practice.holdMs = now - practice.okStartAt;
        const pct = Math.min(100, Math.round((practice.holdMs / HOLD_TO_COMPLETE_MS) * 100));
        document.getElementById('practiceProgressFill').style.width = `${pct}%`;
        document.getElementById('feedbackIcon').textContent = '✅';
        document.getElementById('feedbackText').textContent = 'Nice! Keep holding...';

        // Demo in mockup while holding
        applyMockupEffect(practice.gesture.id, { holding: true });

        if (practice.holdMs >= HOLD_TO_COMPLETE_MS) {
            practice.active = false; // stop counting, but keep live display
            document.getElementById('feedbackIcon').textContent = '🎉';
            document.getElementById('feedbackText').textContent = 'Gesture learned!';
            document.getElementById('practiceStatus').textContent = 'Done. You can use this gesture now.';
            document.getElementById('practiceUseBtn').style.display = '';
            markLearned(practice.gesture.id);
            showToast(`Learned: ${practice.gesture.name}`, 'success');
            applyMockupEffect(practice.gesture.id, { completed: true });
        } else {
            document.getElementById('practiceStatus').textContent = `Hold... ${Math.ceil((HOLD_TO_COMPLETE_MS - practice.holdMs) / 1000)}s`;
        }
    } else {
        practice.okStartAt = 0;
        practice.holdMs = 0;
        document.getElementById('practiceProgressFill').style.width = '0%';
        document.getElementById('feedbackIcon').textContent = '🎯';
        document.getElementById('feedbackText').textContent = 'Adjust your hand to match the required position.';
        document.getElementById('practiceStatus').textContent = `Hold the gesture for ${Math.round(HOLD_TO_COMPLETE_MS / 1000)} seconds...`;
        applyMockupEffect(practice.gesture.id, { holding: false });
    }
}

function useGestureNow() {
    const g = practice.gesture;
    closePracticeModal();

    if (!g) return;
    if (g.mode === 'DRAWING') {
        switchMode('DRAWING');
        switchPage('dashboard');
        showToast('Switched to Drawing mode. Try it live!', 'info');
    } else {
        switchMode('CURSOR');
        switchPage('dashboard');
        showToast('Try it live on the Dashboard (camera feed).', 'info');
    }
}

function clearMockupOverlay() {
    const overlay = document.getElementById('winOverlay');
    if (!overlay) return;
    overlay.classList.remove('show');
    overlay.innerHTML = '';
}

let _mockupEffectTimer = null;
function applyMockupEffect(gestureId, { holding, completed } = {}) {
    const overlay = document.getElementById('winOverlay');
    const cursor = document.getElementById('winCursor');
    const content = document.getElementById('winContentScroll');
    if (!overlay) return;

    if (!holding && !completed) {
        clearMockupOverlay();
        return;
    }

    // Clear any previous effect timer
    if (_mockupEffectTimer) {
        clearTimeout(_mockupEffectTimer);
        _mockupEffectTimer = null;
    }

    overlay.classList.add('show');
    overlay.innerHTML = '';

    const osd = document.createElement('div');
    osd.className = 'win-osd';
    const labelMap = {
        move_cursor: 'Cursor moving',
        pinch_click: 'Left click',
        right_click: 'Right click',
        double_click: 'Double click',
        scroll_up: 'Scroll up',
        scroll_down: 'Scroll down',
        task_view: 'Task View',
        voice_dictation: 'Voice dictation',
        volume: 'Volume',
        draw: 'Drawing',
        brush_size: 'Brush size',
        select_color: 'Color select',
        eraser: 'Eraser',
        clear_canvas: 'Clear canvas',
    };
    osd.innerHTML = `<strong>${labelMap[gestureId] || 'Gesture'}</strong><span>${completed ? '✓' : '…'}</span>`;
    overlay.appendChild(osd);

    // Small gesture-specific visuals
    if (cursor && (gestureId === 'pinch_click' || gestureId === 'double_click' || gestureId === 'right_click')) {
        const rect = cursor.getBoundingClientRect();
        const parentRect = cursor.parentElement.getBoundingClientRect();
        const x = rect.left - parentRect.left + 10;
        const y = rect.top - parentRect.top + 10;

        if (gestureId === 'pinch_click' || gestureId === 'double_click') {
            const ripple = document.createElement('div');
            ripple.className = 'win-click-ripple';
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            cursor.parentElement.appendChild(ripple);
            setTimeout(() => ripple.remove(), 650);
        }
        if (gestureId === 'right_click') {
            const menu = document.createElement('div');
            menu.className = 'win-context-menu';
            menu.style.left = `${x + 40}px`;
            menu.style.top = `${y + 20}px`;
            menu.innerHTML = `
                <div class="win-context-menu-item"><strong>Open</strong><span>↵</span></div>
                <div class="win-context-menu-item"><strong>Copy</strong><span>Ctrl+C</span></div>
                <div class="win-context-menu-item"><strong>Properties</strong><span>Alt+↵</span></div>
            `;
            cursor.parentElement.appendChild(menu);
            setTimeout(() => menu.remove(), 900);
        }
    }

    if (content && gestureId === 'scroll_down') {
        content.scrollTop += 16;
    }
    if (content && gestureId === 'scroll_up') {
        content.scrollTop = Math.max(0, content.scrollTop - 16);
    }

    _mockupEffectTimer = setTimeout(() => {
        clearMockupOverlay();
    }, completed ? 1400 : 500);
}

function resetTutorialProgress() {
    localStorage.removeItem(LEARNED_KEY);
    renderTutorialGestures();
    refreshTutorialCardBadges();
    showToast('Tutorial progress reset — all gestures show "Try it →" again.', 'info');
}

// ── Start/Stop on page visibility ────────────
document.addEventListener('DOMContentLoaded', () => {
    renderTutorialGestures();
    refreshTutorialCardBadges();

    // Ensure we clean up listeners if user closes practice modal by clicking overlay.
    const practiceOverlay = document.getElementById('practiceModal');
    if (practiceOverlay) {
        practiceOverlay.addEventListener('click', (e) => {
            if (e.target === practiceOverlay) {
                stopPractice();
                clearMockupOverlay();
            }
        });
    }

    // Start if tutorial page is already active
    if (currentPage === 'tutorial') {
        startCursorAnimation();
        startTaskbarClock();
    }
});

// Hook into page switching — start/stop animations
const _origSwitchPage = window.switchPage;
if (typeof _origSwitchPage === 'function') {
    window.switchPage = function (page) {
        _origSwitchPage(page);
        if (page === 'tutorial') {
            startCursorAnimation();
            startTaskbarClock();
            renderTutorialGestures();
            refreshTutorialCardBadges();
        } else {
            stopCursorAnimation();
            stopTaskbarClock();
            // Clean up practice listeners if user navigates away mid-practice
            stopPractice();
            clearMockupOverlay();
        }
    };
}