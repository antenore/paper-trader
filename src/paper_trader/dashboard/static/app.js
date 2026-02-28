// Paper Trader — Client-side JS
// HTMX auto-refresh for dashboard partials

document.addEventListener('DOMContentLoaded', () => {
    // Auto-refresh portfolio every 60 seconds on the main dashboard
    const portfolioEl = document.querySelector('[hx-get="/partials/portfolio"]');
    if (portfolioEl) {
        portfolioEl.setAttribute('hx-trigger', 'every 60s');
        htmx.process(portfolioEl);
    }

    // Load scheduler status on dashboard
    if (document.getElementById('scheduler-status')) {
        loadJobStatus();
    }
});

// ── Scheduler job status ─────────────────────────────────────────────

let _pollTimer = null;

async function loadJobStatus() {
    const container = document.getElementById('scheduler-status');
    if (!container) return;

    try {
        const res = await fetch('/api/job-status');
        const jobs = await res.json();
        renderJobCards(container, jobs);

        // Poll while any job is running
        const anyRunning = jobs.some(j => j.is_running);
        if (anyRunning && !_pollTimer) {
            _pollTimer = setInterval(() => loadJobStatus(), 3000);
        } else if (!anyRunning && _pollTimer) {
            clearInterval(_pollTimer);
            _pollTimer = null;
        }
    } catch (e) {
        container.innerHTML = '<p>Failed to load job statuses.</p>';
    }
}

function renderJobCards(container, jobs) {
    container.innerHTML = jobs.map(job => {
        let statusBadge = '';
        if (job.is_running) {
            statusBadge = '<span class="badge" style="background:#dbeafe;color:#1e40af">Running</span>';
        } else if (job.last_status === 'completed') {
            statusBadge = '<span class="badge" style="background:#dcfce7;color:#166534">Done</span>';
        } else if (job.last_status === 'failed') {
            statusBadge = '<span class="badge" style="background:#fee2e2;color:#991b1b">Failed</span>';
        } else {
            statusBadge = '<span class="badge" style="background:#f3f4f6;color:#374151">Idle</span>';
        }

        const lastRun = job.last_started
            ? new Date(job.last_started).toLocaleTimeString()
            : 'Never';

        const disabled = job.is_running ? 'disabled' : '';

        return `
            <div class="stat-card">
                <div class="label">${job.name}</div>
                <div style="margin:0.3rem 0">${statusBadge}</div>
                <div class="label">Last: ${lastRun}</div>
                <button onclick="triggerJob('${job.job_id}')" ${disabled}
                    style="margin-top:0.5rem;padding:0.3rem 0.8rem;font-size:0.8rem"
                    class="outline">${job.is_running ? 'Running...' : 'Run Now'}</button>
            </div>`;
    }).join('');
}

async function triggerJob(jobId) {
    try {
        const res = await fetch(`/trigger/${jobId}`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast(`Started: ${jobId}`);
            // Start polling
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
    toast.style.cssText = `
        position:fixed; bottom:1.5rem; right:1.5rem; padding:0.75rem 1.25rem;
        border-radius:0.5rem; z-index:9999; font-size:0.9rem; font-weight:500;
        color:#fff; background:${isError ? '#ef4444' : '#22c55e'};
        box-shadow:0 4px 12px rgba(0,0,0,0.15);
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}
