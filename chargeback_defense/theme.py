"""
=============================================================================
KAVACH (कवच) - DESIGN SYSTEM & THEME
=============================================================================
DESIGN PLAN:
1. Palette (Premium Dark Armor Theme):
   - Base/Background: Deep Space Navy (#0f172a) to create a premium, high-contrast dark mode.
   - Sidebar: Slightly darker/distinct navy (#0b1120).
   - Accent (Action): Confident Gold/Bronze (#d97706) - used sparingly for primary highlights.
   - Text Colors: Crisp off-white (#f1f5f9) for primary text, cool gray (#94a3b8) for secondary.
   - Semantic Status (Dark mode adjusted):
     * CONTEST (Green): bg #064e3b, fg #34d399
     * REVIEW (Amber): bg #78350f, fg #fbbf24
     * ACCEPT (Red): bg #7f1d1d, fg #f87171
     
2. Typography:
   - 'Outfit' for striking headers, 'Inter' for highly legible body text.
   
3. Layout Concept & Dark Mode Fix:
   - By embracing a full dark mode, we solve the Streamlit default text color clash. The default white text will now sit perfectly on the dark backgrounds.
   - Rigid, structural borders (#334155) keep the "armor/shield" feeling.
=============================================================================
"""

import streamlit as st

def apply_kavach_theme():
    """Injects custom CSS to override Streamlit defaults and apply the Kavach Dark Design System."""
    
    custom_css = """
    <style>
    /* 1. Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;700&display=swap');

    /* 2. Global Variables (Dark Mode) */
    :root {
        --kavach-bg: #0f172a;
        --kavach-sidebar: #0b1120;
        --kavach-gold: #d97706;
        --kavach-border: #334155;
        --kavach-card-bg: #1e293b;
        --kavach-text: #f1f5f9;
        --kavach-text-light: #94a3b8;
        
        --semantic-contest-bg: #064e3b;
        --semantic-contest-fg: #34d399;
        --semantic-review-bg: #78350f;
        --semantic-review-fg: #fbbf24;
        --semantic-accept-bg: #7f1d1d;
        --semantic-accept-fg: #f87171;
    }

    /* 3. Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* 4. Sidebar Custom Elements */
    .kavach-wordmark {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d9488 !important; /* Teal */
        margin-bottom: 0.2rem;
        letter-spacing: 0.05em;
    }
    .kavach-subtitle {
        font-size: 0.9rem;
        color: #9ca3af !important;
        margin-bottom: 2rem;
        font-weight: 600;
    }

    /* 5. Main Content Area & Custom Cards */
    .metric-card {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 4px !important;
        padding: 20px !important;
        margin-bottom: 1rem;
    }
    .metric-card .label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .metric-card .sublabel {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    /* 6. Semantic Badges (Tri-state) */
    .badge-contest, .badge-review, .badge-accept {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .badge-contest { background-color: var(--semantic-contest-bg); color: var(--semantic-contest-fg); }
    .badge-review { background-color: var(--semantic-review-bg); color: var(--semantic-review-fg); }
    .badge-accept { background-color: var(--semantic-accept-bg); color: var(--semantic-accept-fg); }

    /* No more custom overrides for buttons or tabs. Streamlit config.toml handles them natively. */

    /* 9. Hero Moment Shield (Overview Tab only) */
    .hero-shield-container {
        display: flex;
        align-items: center;
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 4px;
    }
    .hero-shield-icon {
        flex: 0 0 80px;
        height: 80px;
        background-color: #111827;
        clip-path: polygon(50% 0%, 100% 0, 100% 70%, 50% 100%, 0 70%, 0 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 2rem;
        position: relative;
        border: 2px solid #374151;
    }
    .hero-shield-icon::after {
        content: '';
        position: absolute;
        width: 60px;
        height: 60px;
        background-color: #0d9488;
        clip-path: polygon(50% 0%, 100% 0, 100% 70%, 50% 100%, 0 70%, 0 0);
    }
    .hero-shield-text h2 {
        margin: 0 0 0.5rem 0 !important;
        font-size: 2rem;
        color: #ffffff !important;
    }
    .hero-shield-text p {
        margin: 0;
        font-size: 1.1rem;
        color: #9ca3af;
    }
    
    /* 10. DataFrames & General Adjustments */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--kavach-border);
    }
    header {
        display: none !important;
    }
    hr {
        border-color: var(--kavach-border) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
