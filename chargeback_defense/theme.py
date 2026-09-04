"""
=============================================================================
KAVACH (कवच) — MISSION CONTROL DESIGN SYSTEM v3.0
=============================================================================
Palette: Deep Charcoal (#0B0F17) base · Emerald (#10b981) / Teal (#0d9488)
         accents · Electric Red (#ef4444) alerts · Gold (#f59e0b) warnings
Typography: Outfit 700 (headers) · Inter 400/600 (body)
Visual: Glassmorphism cards · Animated glow rings · Gradient banners
=============================================================================
"""

import streamlit as st


def apply_kavach_theme() -> None:
    """Inject Kavach Mission Control CSS — FinTech dark mode design system."""
    st.markdown("""
<style>
/* ================================================================
   0. FONT IMPORT
   ================================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ================================================================
   1. GLOBAL RESET & BASE
   ================================================================ */
:root {
    --bg-base:       #0B0F17;
    --bg-surface:    #111827;
    --bg-card:       #141B2D;
    --bg-card-hover: #1a2235;
    --bg-glass:      rgba(20, 27, 45, 0.85);

    --border:        rgba(255,255,255,0.07);
    --border-bright: rgba(255,255,255,0.14);

    --emerald:       #10b981;
    --emerald-dim:   rgba(16, 185, 129, 0.15);
    --emerald-glow:  rgba(16, 185, 129, 0.35);
    --teal:          #0d9488;
    --teal-dim:      rgba(13, 148, 136, 0.15);

    --gold:          #f59e0b;
    --gold-dim:      rgba(245, 158, 11, 0.15);
    --red:           #ef4444;
    --red-dim:       rgba(239, 68, 68, 0.15);
    --red-glow:      rgba(239, 68, 68, 0.4);
    --blue:          #3b82f6;
    --blue-dim:      rgba(59, 130, 246, 0.15);

    --text-primary:  #f0f4f8;
    --text-secondary:#8b9bb4;
    --text-muted:    #4a5568;

    --font-ui:       'Inter', system-ui, sans-serif;
    --font-heading:  'Outfit', sans-serif;
    --font-mono:     'JetBrains Mono', 'Fira Code', monospace;

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  16px;
    --radius-xl:  24px;
}

/* ================================================================
   2. ROOT OVERRIDES — force Streamlit into dark mode
   ================================================================ */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0a0e19 0%, #0d1220 100%) !important;
    border-right: 1px solid var(--border) !important;
}

header[data-testid="stHeader"] { display: none !important; }

/* ================================================================
   3. TYPOGRAPHY
   ================================================================ */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-heading) !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: var(--text-primary) !important;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.65rem !important; }
h3 { font-size: 1.3rem !important; }
code, pre, .font-mono { font-family: var(--font-mono) !important; }

/* ================================================================
   4. GLOBAL SCROLLBAR
   ================================================================ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-surface); }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 2px; }

/* ================================================================
   5. METRIC CARDS (default — dark glass)
   ================================================================ */
.metric-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 20px 22px !important;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--teal), var(--emerald));
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card:hover { border-color: var(--border-bright) !important; }
.metric-card:hover::before { opacity: 1; }
.metric-card .label {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.metric-card .value {
    font-family: var(--font-heading);
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
    line-height: 1.1;
}
.metric-card .sublabel {
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* ================================================================
   6. HERO BANNER (Overview tab top)
   ================================================================ */
.hero-shield-container {
    display: flex;
    align-items: center;
    background: linear-gradient(135deg, #0d1a2e 0%, #0b1f1a 100%);
    border: 1px solid rgba(13,148,136,0.3);
    border-radius: var(--radius-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-shield-container::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-shield-icon {
    flex: 0 0 72px; height: 72px;
    background: linear-gradient(135deg, #0d9488, #10b981);
    clip-path: polygon(50% 0%, 100% 0, 100% 70%, 50% 100%, 0 70%, 0 0);
    margin-right: 2rem;
    box-shadow: 0 0 30px rgba(16,185,129,0.4);
}
.hero-shield-text h2 {
    margin: 0 0 0.4rem 0 !important;
    font-size: 1.9rem !important;
    background: linear-gradient(90deg, #f0f4f8, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-shield-text p { margin: 0; font-size: 1rem; color: var(--text-secondary); }

/* ================================================================
   7. TOP BUSINESS VALUE BANNER CARDS
   ================================================================ */
.banner-card {
    background: linear-gradient(135deg, var(--bg-card), #0d1620);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.banner-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.banner-card .bv-icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
    display: block;
}
.banner-card .bv-value {
    font-family: var(--font-heading);
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--emerald);
    letter-spacing: -0.02em;
    display: block;
    margin-bottom: 4px;
}
.banner-card .bv-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.banner-card.gold .bv-value  { color: var(--gold); }
.banner-card.red  .bv-value  { color: var(--red); }
.banner-card.blue .bv-value  { color: var(--blue); }
.banner-card .bv-glow {
    position: absolute;
    top: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    opacity: 0.12;
    pointer-events: none;
}
.banner-card .bv-glow.green { background: radial-gradient(circle, var(--emerald), transparent); }
.banner-card .bv-glow.gold  { background: radial-gradient(circle, var(--gold), transparent); }
.banner-card .bv-glow.red   { background: radial-gradient(circle, var(--red), transparent); }
.banner-card .bv-glow.blue  { background: radial-gradient(circle, var(--blue), transparent); }

/* ================================================================
   8. SIDEBAR BRANDING
   ================================================================ */
.kavach-wordmark {
    font-family: var(--font-heading);
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #10b981, #0d9488);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
}
.kavach-subtitle {
    font-size: 0.8rem;
    color: var(--text-secondary) !important;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.sidebar-stat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.8rem;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat .stat-val {
    font-family: var(--font-mono);
    color: var(--emerald);
    font-weight: 600;
    font-size: 0.82rem;
}

/* ================================================================
   9. SEMANTIC BADGES
   ================================================================ */
.badge-contest, .badge-review, .badge-accept, .badge-investigate,
.badge-approved, .badge-reserved, .badge-held {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: var(--font-ui);
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.badge-contest    { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-review     { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-accept     { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-investigate{ background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-approved   { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-reserved   { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-held       { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ================================================================
   10. SYNDICATE ALERT BOXES
   ================================================================ */
.syndicate-alert-box {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin: 12px 0;
    position: relative;
    animation: pulseAlert 2s ease-in-out infinite;
}
@keyframes pulseAlert {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 20px 2px rgba(239,68,68,0.15); }
}
.syndicate-alert-box .alert-title {
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 1rem;
    color: #ef4444;
    margin-bottom: 6px;
}
.syndicate-alert-box .alert-body {
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.syndicate-node-pill {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}
.clean-node-pill {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    color: #6ee7b7;
    font-family: var(--font-mono);
    font-size: 0.72px;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
    font-size: 0.72rem;
}
.reserve-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    color: #fde68a;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 3px;
}
.dossier-queued-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 3px;
}

/* ================================================================
   11. NARRATIVE & DOSSIER BOXES
   ================================================================ */
.narrative-box {
    background: linear-gradient(135deg, #0d1a2a, #091420);
    border: 1px solid rgba(13,148,136,0.3);
    border-left: 4px solid var(--teal);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    font-family: var(--font-ui);
    font-size: 0.9rem;
    line-height: 1.75;
    color: #c8d6e5;
    margin: 8px 0;
}
.narrative-missing {
    background: var(--bg-card);
    border: 1px dashed var(--border-bright);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    color: var(--text-secondary);
    font-size: 0.88rem;
}
.sha256-seal {
    background: rgba(16,185,129,0.05);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: #6ee7b7;
    word-break: break-all;
    margin-top: 10px;
}

/* ================================================================
   12. TIMELINE ITEMS
   ================================================================ */
.timeline-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    margin: 3px 0;
    font-size: 0.82rem;
    color: var(--text-muted);
    border-left: 2px solid var(--border);
}
.timeline-item.active {
    border-left-color: var(--teal);
    color: var(--text-secondary);
    background: rgba(13,148,136,0.05);
}

/* ================================================================
   13. STREAMLIT ELEMENT OVERRIDES
   ================================================================ */
/* Buttons */
[data-testid="stButton"] > button {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3) !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-surface) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 6px 16px !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0d9488, #10b981) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px var(--emerald-glow) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* Metrics */
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: var(--font-heading) !important;
    font-weight: 700 !important;
}

/* Dividers */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Info / success / warning / error containers */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}

/* Selectbox / slider labels */
label[data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* Expanders */
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* Code blocks */
[data-testid="stCodeBlock"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)
