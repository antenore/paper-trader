// Paper Trader — Client-side JS
// HTMX auto-refresh for dashboard partials

document.addEventListener('DOMContentLoaded', () => {
    // Auto-refresh portfolio every 60 seconds on the main dashboard
    const portfolioEl = document.querySelector('[hx-get="/partials/portfolio"]');
    if (portfolioEl) {
        portfolioEl.setAttribute('hx-trigger', 'every 60s');
        htmx.process(portfolioEl);
    }
});
