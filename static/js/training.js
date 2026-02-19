/**
 * FG Gesture Control — Training Page Logic
 * Model training with progress ring animation
 */

// ═══════════════════════════════════════════════
// START TRAINING
// ═══════════════════════════════════════════════
async function startTraining() {
    const btn = document.getElementById('trainButton');
    btn.disabled = true;
    btn.textContent = '⏳ Training...';

    // Show progress ring
    const progressContainer = document.getElementById('progressContainer');
    const resultContainer = document.getElementById('trainingResult');
    progressContainer.style.display = '';
    resultContainer.style.display = 'none';

    document.getElementById('trainingHeadline').textContent = 'Training...';
    document.getElementById('trainingMessage').textContent = 'This may take a moment. Do not close the page.';
    document.getElementById('trainingIcon').textContent = '⚙️';

    updateProgressRing(0);

    // Remove stale listeners from any previous training run
    socket.off('training_progress');
    socket.off('training_complete');

    // Listen for progress
    socket.on('training_progress', (data) => {
        updateProgressRing(data.percent);
        document.getElementById('trainingMessage').textContent = data.message || 'Processing...';
    });

    // Listen for completion
    socket.on('training_complete', async (result) => {
        socket.off('training_progress');
        socket.off('training_complete');

        progressContainer.style.display = 'none';

        if (result.success) {
            document.getElementById('trainingIcon').textContent = '⏳';
            document.getElementById('trainingHeadline').textContent = 'Loading Model...';
            document.getElementById('trainingMessage').textContent = 'Reloading the trained model into the engine...';

            // Critical: tell the server to reload the model into the live engine
            try {
                await fetch('/api/model/reload', { method: 'POST' });
            } catch (e) { /* best-effort */ }

            document.getElementById('trainingIcon').textContent = '🎉';
            document.getElementById('trainingHeadline').textContent = 'Training Complete!';
            document.getElementById('trainingMessage').textContent = 'The model is ready to recognize your custom gestures.';

            resultContainer.style.display = '';
            document.getElementById('resultAccuracy').textContent = `${result.accuracy || 0}%`;
            document.getElementById('resultSamples').textContent = result.total_samples || 0;
            document.getElementById('resultGestures').textContent = result.num_classes || 0;

            showToast('Model trained successfully!', 'success');
        } else {
            document.getElementById('trainingIcon').textContent = '❌';
            document.getElementById('trainingHeadline').textContent = 'Training Failed';
            document.getElementById('trainingMessage').textContent = result.error || 'An error occurred during training.';
            showToast('Training failed', 'error');
        }

        btn.disabled = false;
        btn.textContent = '🚀 Train Model';
    });

    // Start training
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.error || 'Failed to start training', 'error');
            btn.disabled = false;
            btn.textContent = '🚀 Train Model';
        }
    } catch (e) {
        showToast('Network error', 'error');
        btn.disabled = false;
        btn.textContent = '🚀 Train Model';
    }
}

// ═══════════════════════════════════════════════
// PROGRESS RING
// ═══════════════════════════════════════════════
function updateProgressRing(percent) {
    const ring = document.getElementById('progressRing');
    const text = document.getElementById('progressText');

    if (!ring || !text) return;

    const circumference = 2 * Math.PI * 54; // radius = 54
    const offset = circumference - (percent / 100) * circumference;

    ring.style.strokeDashoffset = offset;
    text.textContent = `${Math.round(percent)}%`;
}

// ═══════════════════════════════════════════════
// SAMPLE OVERVIEW
// ═══════════════════════════════════════════════
async function loadSampleOverview() {
    try {
        const res = await fetch('/api/gestures');
        const config = await res.json();
        const gestures = config.gestures || [];

        const container = document.getElementById('sampleOverview');
        if (!container) return;

        if (!gestures.length) {
            container.innerHTML = '<div class="sample-empty">No custom gestures created yet. Add gestures first, then record samples.</div>';
            return;
        }

        const totalSamples = gestures.reduce((sum, g) => sum + (g.samples_count || 0), 0);

        const maxSamples = Math.max(...gestures.map(g => g.samples_count || 0), 1);
        container.innerHTML = gestures.map(g => {
            const count = g.samples_count || 0;
            const pct = Math.round((count / maxSamples) * 100);
            return `
            <div class="sample-row">
                <span class="sample-name">${g.name}</span>
                <div class="sample-bar-wrap">
                    <div class="sample-bar-fill" style="width:${pct}%"></div>
                </div>
                <span class="sample-count">${count} samples</span>
            </div>`;
        }).join('');
    } catch (e) {
        // Ignore errors
    }
}

// ── Load on page visit ───────────────────────
document.addEventListener('DOMContentLoaded', () => {
    if (currentPage === 'training') loadSampleOverview();
});