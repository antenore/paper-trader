// Paper Trader — Client-side JS
'use strict';

(function () {

let _pollTimer = null;
let _toastCount = 0;

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('scheduler-status')) {
        loadJobStatus();
    }
});

// ── Scheduler job status ─────────────────────────────────────────────

async function loadJobStatus() {
    const container = document.getElementById('scheduler-status');
    if (!container) return;

    try {
        const res = await fetch('/api/job-status');
        const jobs = await res.json();
        renderJobCards(container, jobs);

        const anyRunning = jobs.some(j => j.is_running);
        if (anyRunning && !_pollTimer) {
            _pollTimer = setInterval(() => loadJobStatus(), 3000);
        } else if (!anyRunning && _pollTimer) {
            clearInterval(_pollTimer);
            _pollTimer = null;
        }
    } catch (e) {
        container.textContent = '';
        const p = document.createElement('p');
        p.textContent = 'Failed to load job statuses.';
        container.appendChild(p);
    }
}

const STATUS_CLASSES = {
    running: 'badge-running',
    completed: 'badge-done',
    failed: 'badge-failed',
};
const STATUS_LABELS = {
    running: 'Running',
    completed: 'Done',
    failed: 'Failed',
};

function renderJobCards(container, jobs) {
    container.textContent = '';
    jobs.forEach(job => {
        const card = document.createElement('div');
        card.className = 'stat-card';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = job.name;
        card.appendChild(label);

        const badgeWrap = document.createElement('div');
        badgeWrap.className = 'badge-status';
        const badge = document.createElement('span');
        const statusKey = job.is_running ? 'running' : (job.last_status || '');
        badge.className = 'badge ' + (STATUS_CLASSES[statusKey] || 'badge-idle');
        badge.textContent = job.is_running ? 'Running' : (STATUS_LABELS[statusKey] || 'Idle');
        badgeWrap.appendChild(badge);
        card.appendChild(badgeWrap);

        const lastRun = document.createElement('div');
        lastRun.className = 'label';
        lastRun.textContent = 'Last: ' + (job.last_started
            ? new Date(job.last_started).toLocaleTimeString()
            : 'Never');
        card.appendChild(lastRun);

        const btn = document.createElement('button');
        btn.className = 'outline btn-sm';
        btn.textContent = job.is_running ? 'Running...' : 'Run Now';
        btn.disabled = job.is_running;
        btn.addEventListener('click', () => triggerJob(job.job_id));
        card.appendChild(btn);

        container.appendChild(card);
    });
}

async function triggerJob(jobId) {
    try {
        const res = await fetch('/trigger/' + encodeURIComponent(jobId), { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast('Started: ' + jobId);
            if (!_pollTimer) {
                _pollTimer = setInterval(() => loadJobStatus(), 3000);
            }
            loadJobStatus();
        } else {
            showToast(data.message, true);
        }
    } catch (e) {
        showToast('Failed to trigger job', true);
    }
}

function showToast(message, isError) {
    const toast = document.createElement('div');
    toast.textContent = message;
    const offset = 1.5 + (_toastCount * 3.5);
    var bg = isError ? 'rgba(248,81,73,0.9)' : 'rgba(63,185,80,0.9)';
    toast.style.cssText =
        'position:fixed; bottom:' + offset + 'rem; right:1.5rem; padding:0.6rem 1.1rem;' +
        'border-radius:6px; z-index:9999; font-size:0.8rem; font-weight:500;' +
        'font-family:DM Mono,monospace; letter-spacing:0.02em;' +
        'color:#fff; background:' + bg + ';' +
        'backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);' +
        'box-shadow:0 8px 24px rgba(0,0,0,0.4); transition:opacity 0.3s, transform 0.3s;' +
        'transform:translateY(0);';
    document.body.appendChild(toast);
    _toastCount++;
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(6px)';
        setTimeout(() => {
            toast.remove();
            _toastCount--;
        }, 300);
    }, 3000);
}

// Expose triggerJob for inline onclick (scheduler cards use addEventListener now,
// but keep as safety net)
window.triggerJob = triggerJob;

})();
