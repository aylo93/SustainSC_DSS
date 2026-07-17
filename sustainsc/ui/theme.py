"""Semantic visual tokens and global CSS for SustainSCM."""

BACKGROUND = "#F4F7F6"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#EDF3F1"
TEXT_PRIMARY = "#17242C"
TEXT_SECONDARY = "#52636D"
BORDER = "#D7E1DE"
PRIMARY = "#087F78"
PRIMARY_HOVER = "#066A65"
ENVIRONMENTAL = "#16846B"
ECONOMIC = "#2F6B9A"
SOCIAL = "#A5632A"
TECHNOLOGICAL = "#6657A6"
SUCCESS = "#287A55"
WARNING = "#A86812"
ERROR = "#B24B4B"
INFO = "#3277A8"

DIMENSION_COLORS = {
    "environmental": ENVIRONMENTAL,
    "economic": ECONOMIC,
    "social": SOCIAL,
    "technological": TECHNOLOGICAL,
}

GLOBAL_CSS = f"""
<style>
:root {{
  --sc-bg: {BACKGROUND}; --sc-surface: {SURFACE}; --sc-muted: {SURFACE_MUTED};
  --sc-text: {TEXT_PRIMARY}; --sc-text-2: {TEXT_SECONDARY}; --sc-border: {BORDER};
  --sc-primary: {PRIMARY}; --sc-primary-hover: {PRIMARY_HOVER};
}}
.stApp {{ background: var(--sc-bg); color: var(--sc-text); }}
[data-testid="stHeader"] {{ background: color-mix(in srgb, var(--sc-bg) 92%, transparent); }}
[data-testid="stSidebar"] {{ background: #EAF1EF; border-right: 1px solid var(--sc-border); }}
.block-container {{ max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ color: var(--sc-text); letter-spacing: -0.02em; }}
h1 {{ font-size: clamp(1.75rem, 3vw, 2.35rem) !important; }}
h2 {{ margin-top: 2.2rem !important; font-size: 1.45rem !important; }}
h3 {{ font-size: 1.08rem !important; }}
p, li, label {{ line-height: 1.55; }}
.sc-page-header {{ border-bottom: 1px solid var(--sc-border); padding: .2rem 0 1.25rem; margin-bottom: 1.25rem; }}
.sc-eyebrow {{ color: var(--sc-primary); font-weight: 700; font-size: .76rem; letter-spacing: .09em; text-transform: uppercase; }}
.sc-page-title {{ margin: .25rem 0 .35rem; font-size: clamp(1.8rem, 3vw, 2.45rem); line-height: 1.12; }}
.sc-page-copy {{ max-width: 760px; color: var(--sc-text-2); font-size: 1rem; }}
.sc-section {{ display:flex; gap:.75rem; align-items:flex-start; margin:2rem 0 .8rem; }}
.sc-section-mark {{ width:4px; min-height:2.4rem; border-radius:4px; background:var(--sc-primary); }}
.sc-section h2 {{ margin:0 !important; }}
.sc-section p {{ margin:.15rem 0 0; color:var(--sc-text-2); }}
.sc-status-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:.8rem 0 1rem; }}
.sc-status-card {{ background:var(--sc-surface); border:1px solid var(--sc-border); border-radius:12px; padding:.85rem 1rem; }}
.sc-status-label {{ color:var(--sc-text-2); font-size:.78rem; font-weight:650; text-transform:uppercase; letter-spacing:.04em; }}
.sc-status-value {{ color:var(--sc-text); font-size:1.35rem; font-weight:720; margin-top:.18rem; }}
.sc-workflow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:.55rem; margin:1rem 0 1.4rem; }}
.sc-stage {{ border:1px solid var(--sc-border); border-radius:10px; padding:.7rem .8rem; background:var(--sc-surface); }}
.sc-stage.complete {{ border-left:4px solid {SUCCESS}; }}
.sc-stage.ready {{ border-left:4px solid {INFO}; }}
.sc-stage.attention {{ border-left:4px solid {WARNING}; }}
.sc-stage.pending {{ border-left:4px solid #9AA8AF; }}
.sc-stage small {{ display:block; color:var(--sc-text-2); margin-top:.2rem; }}
.sc-empty {{ text-align:center; background:var(--sc-surface); border:1px dashed #AFC2BD; border-radius:14px; padding:1.5rem; margin:1rem 0; }}
.sc-empty svg {{ width:52px; height:52px; color:var(--sc-primary); }}
.sc-filter-summary {{ background:#E7F2F1; border:1px solid #B9D7D3; border-radius:9px; padding:.55rem .75rem; color:#285650; font-size:.88rem; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background:var(--sc-primary); border-color:var(--sc-primary); border-radius:8px; font-weight:650;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--sc-primary-hover); }}
[data-testid="stMetric"] {{ background:var(--sc-surface); border:1px solid var(--sc-border); border-radius:11px; padding:.8rem 1rem; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--sc-border); border-radius:10px; overflow:hidden; }}
[data-testid="stAlert"] {{ border-radius:10px; border-width:1px; }}
[data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid var(--sc-border); }}
[data-baseweb="tab"] {{ border-radius:8px 8px 0 0; padding:.65rem .9rem; }}
*:focus-visible {{ outline:3px solid #5FBAB4 !important; outline-offset:2px; }}
@media (max-width: 720px) {{
  .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .sc-status-grid, .sc-workflow {{ grid-template-columns:1fr 1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration:.01ms !important; transition-duration:.01ms !important; }}
}}
</style>
"""
