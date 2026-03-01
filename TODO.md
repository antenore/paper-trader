# Dashboard Frontend Review — TODO

## Alta Priorità

- [x] **#1 — "Total Value" mostra solo il cash**: Calcolato `total_value` in `routes.py` (cash + positions market value) e passato al template.
- [x] **#2 — Classi CSS inesistenti per P&L**: Sostituito `text-success`/`text-danger` con `.positive`/`.negative`.
- [x] **#3 — HTMX auto-refresh è dead code**: Rimosso dead code da `app.js` (il partial portfolio era troppo degradato per sostituire la vista ricca).
- [x] **#5 — Nessuna protezione CSRF**: Aggiunto `csrf.py` con cookie double-submit pattern + `CSRFMiddleware` + hidden input in tutti i form POST.
- [x] **#7 — Status code ignorato nel trigger**: Usato `JSONResponse` con status_code corretto (200/409).

## Media Priorità

- [x] **#8 — Nessuna conferma per azioni distruttive**: Aggiunto `onsubmit="return confirm(...)"` a Go Live, Start Dry Run, Resume Trading.
- [x] **#10 — Journal mostra testo raw**: Aggiunto `marked.js` per rendering markdown nelle entry del journal.
- [x] **#13 — Chart.js caricato su tutte le pagine**: Spostato da `base.html` a `{% block head_scripts %}` solo in `index.html` e `api_usage.html`.
- [x] **#14 — HTMX caricato ovunque**: Rimosso da `base.html` (non più usato attivamente dopo rimozione del dead code HTMX).
- [x] **#15 — Nessun `defer`/`async` sugli script**: Aggiunto `defer` a Chart.js, wrappato chart init in `DOMContentLoaded`.
- [x] **#16 — Import lazy nei route handlers**: Spostati tutti gli import a livello di modulo in `routes.py`.

## Bassa Priorità

- [x] **#4 — Partial degradato**: Aggiornato `partials/portfolio.html` con sector, current price, P&L%, stop-loss + enrichment nel route.
- [x] **#6 — innerHTML con dati dal server**: Riscritto `renderJobCards()` con DOM API (`createElement`/`textContent`) invece di `innerHTML`.
- [x] **#9 — Nessun link attivo nella nav**: Aggiunto `aria-current="page"` con styling CSS `nav a[aria-current="page"]`.
- [x] **#11 — Nessuna paginazione nelle Decisions**: Aggiunto `count_decisions()`, paginazione a 50/pagina con prev/next links.
- [x] **#12 — Toast si sovrappongono**: Toast ora si impilano verticalmente con offset dinamico + fade-out animato.
- [x] **#17 — Colori hardcoded, incompatibili con dark mode**: Tutti i colori in CSS custom properties (`--pt-*`) con variante `[data-theme="dark"]`.
- [x] **#18 — Inline styles sparsi**: Creati classi CSS (`btn-sm`, `btn-inline`, `chart-sm`, `schedule-hour`, `schedule-dow`, `badge-status`).
- [x] **#19 — Hamburger senza `aria-expanded`**: Aggiunto `aria-expanded` toggle dinamico nel `onclick`.
- [x] **#20 — Input senza label nella schedule**: Aggiunto `aria-label` a tutti gli input della schedule table.
- [x] **#21 — Canvas senza fallback**: Aggiunto `role="img"`, `aria-label` e testo fallback a tutti i `<canvas>`.
- [x] **#22 — Fire-and-forget task**: Aggiunto `task.add_done_callback(_log_task_result)` per loggare errori dai task background.
- [x] **#23 — Scope pollution in JS**: Wrappato tutto `app.js` in IIFE con `'use strict'`.
- [x] **#24 — Nessun `preconnect` per i CDN**: Aggiunto `<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>`.
