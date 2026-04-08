"""
URMT — The Ultimate Resilient Modulus Tool (Prediction Module)
Streamlit web application — Polynomial Ridge Regression (Degree 3)
Ported from MATLAB by Mohamad Yaman Fares, Michigan State University
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math, io, csv

# ──────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="URMT — Resilient Modulus Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global */
html, body, [class*="st-"] {
    font-family: 'DM Sans', sans-serif;
}
.block-container { padding-top: 1.5rem; max-width: 1260px; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(78,205,196,.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    margin: 0 0 .35rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -.02em;
}
.hero-banner p {
    margin: 0;
    opacity: .82;
    font-size: .92rem;
    font-weight: 300;
}

/* MR readout card */
.mr-card {
    background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    color: #fff;
    text-align: center;
    margin: .5rem 0;
}
.mr-card .mr-label { font-size: .82rem; opacity: .7; margin-bottom: .25rem; }
.mr-card .mr-value { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.mr-card .mr-unit  { font-size: .85rem; opacity: .6; }

/* Status lamps */
.lamp { display: inline-block; width: 13px; height: 13px; border-radius: 50%; margin-left: 6px; vertical-align: middle; box-shadow: 0 0 4px rgba(0,0,0,.15); }
.lamp-green  { background: #2ecc71; box-shadow: 0 0 6px rgba(46,204,113,.45); }
.lamp-yellow { background: #f1c40f; box-shadow: 0 0 6px rgba(241,196,15,.45); }
.lamp-red    { background: #e74c3c; box-shadow: 0 0 6px rgba(231,76,60,.45); }
.lamp-gray   { background: #95a5a6; }

/* Validation banners */
.valid-ok   { color: #27ae60; font-weight: 600; font-size: .88rem; }
.valid-fail { color: #e74c3c; font-weight: 600; font-size: .88rem; }

/* Section headings */
.section-head {
    background: linear-gradient(90deg, #1a3a4a 0%, #2c5364 100%);
    color: #fff;
    padding: .55rem 1rem;
    border-radius: 8px;
    font-size: .95rem;
    font-weight: 600;
    margin: 1.2rem 0 .7rem 0;
    letter-spacing: .01em;
}

/* Sensitivity bar */
.sens-pos { color: #27ae60; font-weight: 600; }
.sens-neg { color: #e74c3c; font-weight: 600; }

/* Footer */
.footer {
    text-align: center;
    color: #7f8c8d;
    font-size: .78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #ecf0f1;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
#  POLYNOMIAL MODELS  (auto-converted from MATLAB)
# ──────────────────────────────────────────────────────────────────────
from models import poly_base_subbase, poly_subgrade

def compute_mr(model_name: str, x: list) -> float:
    """Compute MR (MPa) for given model and 12 input features (SI)."""
    if model_name == "Base/Subbase":
        return poly_base_subbase(x)
    else:
        return poly_subgrade(x)

# ──────────────────────────────────────────────────────────────────────
#  CONSTANTS & LIMITS
# ──────────────────────────────────────────────────────────────────────
NEAR_EDGE_FRAC = 0.10

LIMITS_SI = {
    "Base/Subbase": np.array([
        [39.46, 64.00], [31.00, 50.41], [5.00, 7.58],
        [5.44, 18.00], [0.78, 2.50], [0.13, 0.23],
        [0.00, 16.64], [0.00, 12.50],
        [21.65, 22.96], [4.40, 7.01],
        [7.85, 164.87], [76.66, 763.75],
    ]),
    "Subgrade": np.array([
        [1.00, 4.17], [23.84, 45.13], [52.16, 74.85],
        [0.06, 0.21], [0.03, 0.06], [0.01, 0.01],
        [18.29, 24.70], [2.07, 11.99],
        [17.69, 20.70], [9.80, 14.50],
        [11.13, 41.35], [92.00, 210.16],
    ]),
}

MR_LIMITS_SI = {
    "Base/Subbase": [61.35, 512.70],
    "Subgrade": [78.15, 266.67],
}

MEDIANS_SI = {
    "Base/Subbase": [53.66, 38.76, 6.49, 8.19, 1.58, 0.19, 0.00, 0.00, 22.30, 5.20, 46.67, 343.00],
    "Subgrade": [2.70, 37.50, 57.76, 0.11, 0.04, 0.01, 22.03, 7.64, 19.79, 11.60, 16.84, 118.10],
}

PARAM_NAMES = [
    "Gravel (%)", "Sand (%)", "Fines (%)",
    "D60", "D30", "D10",
    "LL (%)", "PL (%)",
    "MDU", "OMC (%)",
    "τ_oct", "θ",
]

# AASHTO T307 stress sequences (kPa)
AASHTO_BASE = {
    "s3": [20.7, 20.7, 20.7, 34.5, 34.5, 34.5, 68.9, 68.9, 68.9, 103.4, 103.4, 103.4, 137.9, 137.9, 137.9],
    "sd": [20.7, 41.4, 62.1, 34.5, 68.9, 103.4, 68.9, 137.9, 206.8, 68.9, 103.4, 206.8, 103.4, 137.9, 275.8],
}
AASHTO_SUB = {
    "s3": [41.4]*5 + [27.6]*5 + [13.8]*5,
    "sd": [13.8, 27.6, 41.4, 55.2, 68.9]*3,
}

# ──────────────────────────────────────────────────────────────────────
#  UNIT CONVERSIONS
# ──────────────────────────────────────────────────────────────────────
PSI_PER_KPA = 0.1450377377
PCF_PER_KNM3 = 6.365880354264158

def to_si_len(v, u):    return v * 25.4 if u == "US" else v
def to_si_stress(v, u): return v / PSI_PER_KPA if u == "US" else v
def to_si_den(v, u):    return v / PCF_PER_KNM3 if u == "US" else v
def from_si_len(v, u):    return v / 25.4 if u == "US" else v
def from_si_stress(v, u): return v * PSI_PER_KPA if u == "US" else v
def from_si_den(v, u):    return v * PCF_PER_KNM3 if u == "US" else v
def mpa_to_ksi(v): return v * PSI_PER_KPA

def convert_limits(L_SI, u):
    L = L_SI.copy()
    if u == "US":
        L[3:6] = L[3:6] / 25.4
        L[8]   = L[8] * PCF_PER_KNM3
        L[10:12] = L[10:12] * PSI_PER_KPA
    return L

# ──────────────────────────────────────────────────────────────────────
#  LAMP STATUS
# ──────────────────────────────────────────────────────────────────────
def lamp_status(v, lo, hi):
    """Return 'green', 'yellow', 'red', or 'gray'."""
    if not np.isfinite(v):
        return "red"
    if v < lo or v > hi:
        return "red"
    rng = max(hi - lo, 1e-12)
    band = NEAR_EDGE_FRAC * rng
    if v <= lo + band or v >= hi - band:
        return "yellow"
    return "green"

def lamp_html(status):
    return f'<span class="lamp lamp-{status}"></span>'

# ──────────────────────────────────────────────────────────────────────
#  UI LABELS
# ──────────────────────────────────────────────────────────────────────
def param_labels(u):
    lu = "in" if u == "US" else "mm"
    du = "pcf" if u == "US" else "kN/m³"
    su = "psi" if u == "US" else "kPa"
    return [
        "Gravel (%)", "Sand (%)", "Fines (%)",
        f"D60 ({lu})", f"D30 ({lu})", f"D10 ({lu})",
        "LL (%)", "PL (%)",
        f"MDU ({du})", "OMC (%)",
        f"Octahedral Shear τ ({su})", f"Bulk Stress θ ({su})",
    ]

# ──────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────
if "pred_data" not in st.session_state:
    st.session_state.pred_data = None
    st.session_state.sens_data = None
    st.session_state.mr_value = None
    st.session_state.last_x_si = None
    st.session_state.last_s3 = None
    st.session_state.last_sd = None

# ──────────────────────────────────────────────────────────────────────
#  HERO BANNER
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🔬 The Ultimate Resilient Modulus Tool</h1>
    <p>Polynomial Ridge Regression (Degree 3) — MR Prediction for Base/Subbase & Subgrade Materials</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
#  SIDEBAR — Controls
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    units = st.radio("**Unit System**", ["SI", "US"],
                     help="SI → mm, kN/m³, kPa, MPa  |  US → in, pcf, psi, ksi",
                     horizontal=True)
    model = st.selectbox("**Material Model**", ["Base/Subbase", "Subgrade"])

    st.divider()
    st.markdown("### 🎯 Quick Actions")
    load_defaults = st.button("📋 Load Median Defaults", use_container_width=True)
    clear_all = st.button("🗑️ Clear All", use_container_width=True)

    st.divider()
    st.markdown("### 📊 Chart X-Axis")
    x_axis_choice = st.radio("Select X-Axis", ["Sequence Number", "Bulk Stress", "Octahedral Shear Stress"], label_visibility="collapsed")

    st.divider()
    st.caption("URMT v2.0 — © 2025 Michigan State University")
    st.caption("Author: Mohamad Yaman Fares")

# ──────────────────────────────────────────────────────────────────────
#  INPUT FIELDS
# ──────────────────────────────────────────────────────────────────────
labels = param_labels(units)
limits = convert_limits(LIMITS_SI[model], units)

# Get defaults
if load_defaults:
    meds = MEDIANS_SI[model].copy()
    defaults = meds.copy()
    # Convert from SI to display units
    defaults[3] = from_si_len(meds[3], units)
    defaults[4] = from_si_len(meds[4], units)
    defaults[5] = from_si_len(meds[5], units)
    defaults[8] = from_si_den(meds[8], units)
    defaults[10] = from_si_stress(meds[10], units)
    defaults[11] = from_si_stress(meds[11], units)
    st.session_state._defaults = defaults
elif clear_all:
    st.session_state._defaults = [0.0]*12
    st.session_state.pred_data = None
    st.session_state.sens_data = None
    st.session_state.mr_value = None

defaults = st.session_state.get("_defaults", [0.0]*12)

st.markdown('<div class="section-head">📝 Input Parameters — Soil Index Properties & Stress State</div>', unsafe_allow_html=True)

# Row 1: Gravel, Sand, Fines, LL
vals = [0.0] * 12
c1, c2, c3, c4 = st.columns(4)
with c1:
    vals[0] = st.number_input(labels[0], value=defaults[0], format="%.4f", key="x1")
with c2:
    vals[1] = st.number_input(labels[1], value=defaults[1], format="%.4f", key="x2")
with c3:
    vals[2] = st.number_input(labels[2], value=defaults[2], format="%.4f", key="x3")
with c4:
    vals[6] = st.number_input(labels[6], value=defaults[6], format="%.4f", key="x7")

# Row 2: D60, D30, D10, PL
c5, c6, c7, c8 = st.columns(4)
with c5:
    vals[3] = st.number_input(labels[3], value=defaults[3], format="%.4f", key="x4")
with c6:
    vals[4] = st.number_input(labels[4], value=defaults[4], format="%.4f", key="x5")
with c7:
    vals[5] = st.number_input(labels[5], value=defaults[5], format="%.4f", key="x6")
with c8:
    vals[7] = st.number_input(labels[7], value=defaults[7], format="%.4f", key="x8")

# Row 3: MDU, OMC, tau, theta
c9, c10, c11, c12 = st.columns(4)
with c9:
    vals[8] = st.number_input(labels[8], value=defaults[8], format="%.4f", key="x9")
with c10:
    vals[9] = st.number_input(labels[9], value=defaults[9], format="%.4f", key="x10")
with c11:
    vals[10] = st.number_input(labels[10], value=defaults[10], format="%.4f", key="x11")
with c12:
    vals[11] = st.number_input(labels[11], value=defaults[11], format="%.4f", key="x12")

# ──────────────────────────────────────────────────────────────────────
#  LIVE VALIDATION
# ──────────────────────────────────────────────────────────────────────
vc1, vc2 = st.columns(2)

# Sum check
grain_sum = vals[0] + vals[1] + vals[2]
with vc1:
    if abs(grain_sum - 100) < 0.5:
        st.markdown(f'<span class="valid-ok">✓ Gravel + Sand + Fines = {grain_sum:.1f}%</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="valid-fail">✗ Gravel + Sand + Fines = {grain_sum:.1f}% (must = 100%)</span>', unsafe_allow_html=True)

# Grain size order
d10_si = to_si_len(vals[5], units)
d30_si = to_si_len(vals[4], units)
d60_si = to_si_len(vals[3], units)
with vc2:
    if d10_si <= d30_si and d30_si <= d60_si:
        st.markdown('<span class="valid-ok">✓ Grain size order: D10 ≤ D30 ≤ D60</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="valid-fail">✗ Grain size order INVALID (need D10 ≤ D30 ≤ D60)</span>', unsafe_allow_html=True)

# Lamp status for each input
statuses = []
for i in range(12):
    s = lamp_status(vals[i], limits[i, 0], limits[i, 1])
    statuses.append(s)

# Show lamps row
lamp_cols = st.columns(12)
for i, lc in enumerate(lamp_cols):
    with lc:
        short = PARAM_NAMES[i]
        st.markdown(f'{short} {lamp_html(statuses[i])}', unsafe_allow_html=True)

has_red = any(s == "red" for s in statuses)
if has_red:
    red_params = [PARAM_NAMES[i] for i in range(12) if statuses[i] == "red"]
    st.warning(f"⚠️ Out-of-range inputs detected: **{', '.join(red_params)}**. Prediction will be blocked.")

# ──────────────────────────────────────────────────────────────────────
#  PREDICT BUTTON
# ──────────────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🚀 **Predict Resilient Modulus**", type="primary", use_container_width=True)

if predict_btn:
    if has_red:
        st.error("❌ Prediction blocked — one or more inputs fall outside the Q25–Q75 training data bounds. Adjust highlighted inputs.")
    else:
        # Convert to SI
        x_SI = [0.0] * 12
        x_SI[0] = vals[0]; x_SI[1] = vals[1]; x_SI[2] = vals[2]
        x_SI[3] = to_si_len(vals[3], units)
        x_SI[4] = to_si_len(vals[4], units)
        x_SI[5] = to_si_len(vals[5], units)
        x_SI[6] = vals[6]; x_SI[7] = vals[7]
        x_SI[8] = to_si_den(vals[8], units)
        x_SI[9] = vals[9]
        x_SI[10] = to_si_stress(vals[10], units)
        x_SI[11] = to_si_stress(vals[11], units)

        # Single MR prediction
        mr_mpa = compute_mr(model, x_SI)

        # AASHTO T307 sequences
        seq_data = AASHTO_BASE if model == "Base/Subbase" else AASHTO_SUB
        s3_list = seq_data["s3"]
        sd_list = seq_data["sd"]

        rows = []
        for i in range(15):
            theta_kPa = sd_list[i] + 3 * s3_list[i]
            tau_kPa = (math.sqrt(2) / 3) * sd_list[i]
            xi = x_SI.copy()
            xi[10] = tau_kPa
            xi[11] = theta_kPa
            mr_i = compute_mr(model, xi)

            if units == "US":
                rows.append([model, i+1, from_si_stress(theta_kPa, "US"), from_si_stress(tau_kPa, "US"), mpa_to_ksi(mr_i)])
            else:
                rows.append([model, i+1, theta_kPa, tau_kPa, mr_i])

        stress_u = "psi" if units == "US" else "kPa"
        mr_u = "ksi" if units == "US" else "MPa"

        df = pd.DataFrame(rows, columns=["Model", "Seq", f"Bulk Stress ({stress_u})", f"Oct. Shear Stress ({stress_u})", f"MR ({mr_u})"])

        # Sensitivity analysis
        sens_rows = []
        mr_base = compute_mr(model, x_SI)
        mr_base_d = mpa_to_ksi(mr_base) if units == "US" else mr_base
        for i in range(12):
            xp = x_SI.copy(); xp[i] *= 1.10
            xm = x_SI.copy(); xm[i] *= 0.90
            mrp = compute_mr(model, xp)
            mrm = compute_mr(model, xm)
            mrp_d = mpa_to_ksi(mrp) if units == "US" else mrp
            mrm_d = mpa_to_ksi(mrm) if units == "US" else mrm
            sens_rows.append([
                PARAM_NAMES[i],
                round(mr_base_d, 4),
                round(mrp_d, 4),
                round(mrm_d, 4),
                round(mrp_d - mr_base_d, 4),
                round(mrm_d - mr_base_d, 4),
            ])
        sens_df = pd.DataFrame(sens_rows, columns=["Parameter", f"Baseline MR ({mr_u})", f"+10% MR ({mr_u})", f"−10% MR ({mr_u})", f"Δ+ ({mr_u})", f"Δ− ({mr_u})"])

        # Store in session
        st.session_state.pred_data = df
        st.session_state.sens_data = sens_df
        st.session_state.mr_value = mpa_to_ksi(mr_mpa) if units == "US" else mr_mpa
        st.session_state.mr_unit = mr_u
        st.session_state.last_x_si = x_SI
        st.session_state.last_s3 = s3_list
        st.session_state.last_sd = sd_list
        st.session_state.last_model = model
        st.session_state.last_units = units

# ──────────────────────────────────────────────────────────────────────
#  RESULTS DISPLAY
# ──────────────────────────────────────────────────────────────────────
if st.session_state.mr_value is not None:
    mr_u = st.session_state.mr_unit
    mr_v = st.session_state.mr_value

    # MR readout
    st.markdown(f"""
    <div class="mr-card">
        <div class="mr-label">Predicted Resilient Modulus</div>
        <div class="mr-value">{mr_v:.3f}</div>
        <div class="mr-unit">{mr_u}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart ──
    st.markdown('<div class="section-head">📈 MR vs. Stress Sequence — AASHTO T 307</div>', unsafe_allow_html=True)

    df = st.session_state.pred_data
    u = st.session_state.last_units
    s3_list = st.session_state.last_s3
    sd_list = st.session_state.last_sd

    stress_u = "psi" if u == "US" else "kPa"
    mr_col = df.columns[-1]

    if x_axis_choice == "Sequence Number":
        x_data = df["Seq"]
        x_label = "Sequence Number"
    elif x_axis_choice == "Bulk Stress":
        x_data = df.iloc[:, 2]
        x_label = f"Bulk Stress θ ({stress_u})"
    else:
        x_data = df.iloc[:, 3]
        x_label = f"Octahedral Shear Stress τ ({stress_u})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data, y=df[mr_col],
        mode='lines+markers+text',
        marker=dict(size=10, color='#2c5364', line=dict(width=1.5, color='white')),
        line=dict(width=2.5, color='#2c5364'),
        text=[str(i) for i in range(1, 16)],
        textposition="top center",
        textfont=dict(size=9, color='#7f8c8d'),
        hovertemplate=f'{x_label}: %{{x:.2f}}<br>MR: %{{y:.3f}} {mr_u}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=f"Resilient Modulus ({mr_u})",
        title=dict(text=f"{st.session_state.last_model} — AASHTO T307 Sequences", font=dict(size=16)),
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=30, t=50, b=50),
        font=dict(family="DM Sans"),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#ecf0f1')
    fig.update_yaxes(showgrid=True, gridcolor='#ecf0f1')
    st.plotly_chart(fig, use_container_width=True)

    # ── Results Table ──
    st.markdown('<div class="section-head">📋 AASHTO T307 Prediction Results</div>', unsafe_allow_html=True)

    # Flag rows where stress is outside training bounds
    lim = LIMITS_SI[st.session_state.last_model]
    tau_lo, tau_hi = lim[10, 0], lim[10, 1]
    theta_lo, theta_hi = lim[11, 0], lim[11, 1]

    def highlight_oor(row):
        idx = int(row["Seq"]) - 1
        theta_i = sd_list[idx] + 3 * s3_list[idx]
        tau_i = (math.sqrt(2)/3) * sd_list[idx]
        if theta_i < theta_lo or theta_i > theta_hi or tau_i < tau_lo or tau_i > tau_hi:
            return ['background-color: #fde8e8'] * len(row)
        return [''] * len(row)

    styled = df.style.apply(highlight_oor, axis=1).format(precision=3)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("* Rows highlighted in red indicate stress states outside the Q25–Q75 training data bounds.")

    # ── Sensitivity Analysis ──
    st.markdown('<div class="section-head">🔍 Sensitivity Analysis (±10% perturbation)</div>', unsafe_allow_html=True)
    sens_df = st.session_state.sens_data

    def color_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #27ae60; font-weight: 600'
            elif val < 0:
                return 'color: #e74c3c; font-weight: 600'
        return ''

    delta_cols = [c for c in sens_df.columns if "Δ" in c]
    styled_sens = sens_df.style.map(color_delta, subset=delta_cols).format(precision=4)
    st.dataframe(styled_sens, use_container_width=True, hide_index=True)

    # ── Export ──
    st.markdown('<div class="section-head">💾 Export</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.write("\n\nSensitivity Analysis\n")
        sens_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv_buf.getvalue(),
            file_name="URMT_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        st.info("💡 For PDF reports, use the original MATLAB tool or print this page (Ctrl+P).")

# ──────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    URMT v2.0 — Polynomial Ridge Regression (Deg. 3) — © 2025 Michigan State University<br>
    Author: Mohamad Yaman Fares · Civil & Environmental Engineering · faresmoh@msu.edu
</div>
""", unsafe_allow_html=True)
