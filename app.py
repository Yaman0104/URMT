"""
URMT - The Ultimate Resilient Modulus Tool (Prediction Module)
Streamlit web application - Polynomial Ridge Regression (Degree 3)
Ported from MATLAB by Mohamad Yaman Fares, Michigan State University
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import io

# =====================================================================
#  PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="URMT - Resilient Modulus Prediction",
    page_icon="https://em-content.zobj.net/source/twitter/376/microscope_1f52c.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
#  CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
    color: #fff;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; right: -8%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(78,205,196,.16) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 { margin: 0 0 .3rem 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -.02em; }
.hero-banner p  { margin: 0; opacity: .78; font-size: .88rem; font-weight: 300; }

/* MR readout */
.mr-card {
    background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
    border-radius: 12px; padding: 1.2rem 1.6rem;
    color: #fff; text-align: center; margin: .4rem 0;
}
.mr-card .mr-label { font-size: .8rem; opacity: .65; margin-bottom: .2rem; }
.mr-card .mr-value { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.mr-card .mr-unit  { font-size: .82rem; opacity: .55; }

/* Status indicators */
.lamp {
    display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
}
.lamp-green  { background: #2ecc71; box-shadow: 0 0 5px rgba(46,204,113,.4); }
.lamp-yellow { background: #f39c12; box-shadow: 0 0 5px rgba(243,156,18,.4); }
.lamp-red    { background: #e74c3c; box-shadow: 0 0 5px rgba(231,76,60,.4); }
.lamp-gray   { background: #95a5a6; }

/* Input card with range */
.input-card {
    background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px;
    padding: .55rem .7rem; margin-bottom: .3rem; font-size: .78rem; color: #555;
}
.input-card .range-label { font-weight: 600; color: #2c5364; }

/* Validation */
.valid-ok   { color: #27ae60; font-weight: 600; font-size: .86rem; }
.valid-fail { color: #e74c3c; font-weight: 600; font-size: .86rem; }

/* Section heads */
.section-head {
    background: linear-gradient(90deg, #1a3a4a 0%, #2c5364 100%);
    color: #fff; padding: .5rem .9rem; border-radius: 8px;
    font-size: .9rem; font-weight: 600; margin: 1rem 0 .6rem 0;
}

/* Footer */
.footer {
    text-align: center; color: #7f8c8d; font-size: .76rem;
    padding: 1.8rem 0 .8rem 0; border-top: 1px solid #ecf0f1; margin-top: 1.5rem;
}

/* Tweak number input label size */
div[data-testid="stNumberInput"] label p { font-size: .84rem !important; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
#  POLYNOMIAL MODELS
# =====================================================================
from models import poly_base_subbase, poly_subgrade

def compute_mr(model_name: str, x: list) -> float:
    if model_name == "Base/Subbase":
        return poly_base_subbase(x)
    else:
        return poly_subgrade(x)

# =====================================================================
#  CONSTANTS & LIMITS
# =====================================================================
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
    "Subgrade":     [2.70, 37.50, 57.76, 0.11, 0.04, 0.01, 22.03, 7.64, 19.79, 11.60, 16.84, 118.10],
}

PARAM_SHORT = [
    "Gravel", "Sand", "Fines", "D60", "D30", "D10",
    "LL", "PL", "MDU", "OMC", "tau_oct", "theta",
]

AASHTO_BASE = {
    "s3": [20.7, 20.7, 20.7, 34.5, 34.5, 34.5, 68.9, 68.9, 68.9, 103.4, 103.4, 103.4, 137.9, 137.9, 137.9],
    "sd": [20.7, 41.4, 62.1, 34.5, 68.9, 103.4, 68.9, 137.9, 206.8, 68.9, 103.4, 206.8, 103.4, 137.9, 275.8],
}
AASHTO_SUB = {
    "s3": [41.4]*5 + [27.6]*5 + [13.8]*5,
    "sd": [13.8, 27.6, 41.4, 55.2, 68.9]*3,
}

INPUT_KEYS = [f"inp_{i}" for i in range(12)]

# =====================================================================
#  UNIT CONVERSIONS
# =====================================================================
PSI_PER_KPA = 0.1450377377
PCF_PER_KNM3 = 6.365880354264158

def to_si_len(v, u):      return v * 25.4 if u == "US" else v
def to_si_stress(v, u):   return v / PSI_PER_KPA if u == "US" else v
def to_si_den(v, u):      return v / PCF_PER_KNM3 if u == "US" else v
def from_si_len(v, u):    return v / 25.4 if u == "US" else v
def from_si_stress(v, u): return v * PSI_PER_KPA if u == "US" else v
def from_si_den(v, u):    return v * PCF_PER_KNM3 if u == "US" else v
def mpa_to_ksi(v):        return v * PSI_PER_KPA

def convert_limits(L_SI, u):
    L = L_SI.copy()
    if u == "US":
        L[3:6]   = L[3:6] / 25.4
        L[8]     = L[8] * PCF_PER_KNM3
        L[10:12] = L[10:12] * PSI_PER_KPA
    return L

def convert_si_to_display(vals_si, u):
    d = list(vals_si)
    d[3] = from_si_len(d[3], u)
    d[4] = from_si_len(d[4], u)
    d[5] = from_si_len(d[5], u)
    d[8] = from_si_den(d[8], u)
    d[10] = from_si_stress(d[10], u)
    d[11] = from_si_stress(d[11], u)
    return d

def convert_display_to_si(vals_disp, u):
    s = list(vals_disp)
    s[3] = to_si_len(s[3], u)
    s[4] = to_si_len(s[4], u)
    s[5] = to_si_len(s[5], u)
    s[8] = to_si_den(s[8], u)
    s[10] = to_si_stress(s[10], u)
    s[11] = to_si_stress(s[11], u)
    return s

# =====================================================================
#  LAMP STATUS
# =====================================================================
def lamp_status(v, lo, hi):
    if not np.isfinite(v):
        return "red"
    if lo == hi:
        return "green" if abs(v - lo) < 1e-6 else "red"
    if v < lo or v > hi:
        return "red"
    rng = hi - lo
    band = NEAR_EDGE_FRAC * rng
    if v <= lo + band or v >= hi - band:
        return "yellow"
    return "green"

def lamp_html(status):
    return f'<span class="lamp lamp-{status}"></span>'

# =====================================================================
#  UI LABELS
# =====================================================================
def param_labels(u):
    lu = "in" if u == "US" else "mm"
    du = "pcf" if u == "US" else "kN/m3"
    su = "psi" if u == "US" else "kPa"
    return [
        "Gravel (%)", "Sand (%)", "Fines (%)",
        f"D60 ({lu})", f"D30 ({lu})", f"D10 ({lu})",
        "LL (%)", "PL (%)",
        f"MDU ({du})", "OMC (%)",
        f"Octahedral Shear Stress ({su})", f"Bulk Stress ({su})",
    ]

# =====================================================================
#  SESSION STATE INIT
# =====================================================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.pred_data = None
    st.session_state.sens_data = None
    st.session_state.mr_value = None
    st.session_state.mr_unit = "MPa"
    st.session_state.last_model = None
    st.session_state.last_units = "SI"
    st.session_state.last_s3 = None
    st.session_state.last_sd = None
    for k in INPUT_KEYS:
        st.session_state[k] = 0.0

# =====================================================================
#  HERO BANNER
# =====================================================================
st.markdown("""
<div class="hero-banner">
    <h1>The Ultimate Resilient Modulus Tool</h1>
    <p>Polynomial Ridge Regression (Degree 3) &mdash; MR Prediction for Base/Subbase &amp; Subgrade Materials</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Settings")
    units = st.radio("Unit System", ["SI", "US"],
                     help="SI: mm, kN/m3, kPa, MPa  |  US: in, pcf, psi, ksi",
                     horizontal=True)
    model = st.selectbox("Material Model", ["Base/Subbase", "Subgrade"])

    st.divider()
    st.markdown("### Quick Actions")
    load_defaults = st.button("Load Median Defaults", use_container_width=True)
    clear_all = st.button("Clear All Inputs", use_container_width=True)

    st.divider()
    st.markdown("### Chart X-Axis")
    x_axis_choice = st.radio(
        "x_axis_sel",
        ["Sequence Number", "Bulk Stress", "Octahedral Shear Stress"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### About")
    st.caption("URMT v2.0 | Poly. Ridge Regression (Deg. 3)")
    st.caption("(c) 2025 Michigan State University")
    st.caption("Author: Mohamad Yaman Fares")

# =====================================================================
#  HANDLE LOAD DEFAULTS / CLEAR
# =====================================================================
if load_defaults:
    meds_si = list(MEDIANS_SI[model])
    meds_disp = convert_si_to_display(meds_si, units)
    for i, k in enumerate(INPUT_KEYS):
        st.session_state[k] = float(meds_disp[i])
    st.rerun()

if clear_all:
    for k in INPUT_KEYS:
        st.session_state[k] = 0.0
    st.session_state.pred_data = None
    st.session_state.sens_data = None
    st.session_state.mr_value = None
    st.rerun()

# =====================================================================
#  COMPUTE LIMITS IN DISPLAY UNITS
# =====================================================================
labels = param_labels(units)
limits = convert_limits(LIMITS_SI[model], units)

# =====================================================================
#  INPUT FIELDS  (4 columns x 3 rows, each with range card)
# =====================================================================
st.markdown('<div class="section-head">Input Parameters - Soil Index Properties and Stress State</div>', unsafe_allow_html=True)

INPUT_LAYOUT = [
    [0, 1, 2, 6],
    [3, 4, 5, 7],
    [8, 9, 10, 11],
]

vals = [0.0] * 12

for row_indices in INPUT_LAYOUT:
    cols = st.columns(4)
    for col_idx, param_idx in enumerate(row_indices):
        with cols[col_idx]:
            lo = limits[param_idx, 0]
            hi = limits[param_idx, 1]
            st.markdown(
                f'<div class="input-card">'
                f'<span class="range-label">Valid Q25-Q75:</span> '
                f'{lo:.4g} &ndash; {hi:.4g}'
                f'</div>',
                unsafe_allow_html=True,
            )
            v = st.number_input(
                labels[param_idx],
                format="%.4f",
                key=INPUT_KEYS[param_idx],
                step=None,
            )
            vals[param_idx] = v if v is not None else 0.0

# =====================================================================
#  LIVE VALIDATION
# =====================================================================
st.markdown("")
vc1, vc2 = st.columns(2)

grain_sum = vals[0] + vals[1] + vals[2]
with vc1:
    if abs(grain_sum - 100) < 0.5:
        st.markdown(
            f'<span class="valid-ok">&#10003; Gravel + Sand + Fines = {grain_sum:.1f}%</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="valid-fail">&#10007; Gravel + Sand + Fines = {grain_sum:.1f}% (must equal 100%)</span>',
            unsafe_allow_html=True,
        )

d10_si = to_si_len(vals[5], units)
d30_si = to_si_len(vals[4], units)
d60_si = to_si_len(vals[3], units)
with vc2:
    if d10_si <= d30_si and d30_si <= d60_si:
        st.markdown(
            '<span class="valid-ok">&#10003; Grain size order: D10 &le; D30 &le; D60</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="valid-fail">&#10007; Grain size order INVALID (need D10 &le; D30 &le; D60)</span>',
            unsafe_allow_html=True,
        )

# Lamp statuses
statuses = [lamp_status(vals[i], limits[i, 0], limits[i, 1]) for i in range(12)]

status_labels = {
    "green": "In range",
    "yellow": "Near edge",
    "red": "OUT OF RANGE",
}

lamp_row_html = '<div style="display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 12px 0;">'
for i in range(12):
    lo = limits[i, 0]
    hi = limits[i, 1]
    color = statuses[i]
    tooltip = f"{PARAM_SHORT[i]}: {status_labels[color]} [{lo:.4g}, {hi:.4g}]"
    lamp_row_html += (
        f'<div style="font-size:.8rem; color:#555;" title="{tooltip}">'
        f'{lamp_html(color)} {PARAM_SHORT[i]}'
        f'</div>'
    )
lamp_row_html += '</div>'
st.markdown(lamp_row_html, unsafe_allow_html=True)

has_red = any(s == "red" for s in statuses)
if has_red:
    red_names = [PARAM_SHORT[i] for i in range(12) if statuses[i] == "red"]
    st.error(
        f"Out-of-range inputs: **{', '.join(red_names)}** "
        f"-- prediction blocked until these are within Q25-Q75 training bounds."
    )

# =====================================================================
#  PREDICT BUTTON
# =====================================================================
st.markdown("")
predict_btn = st.button("Predict Resilient Modulus", type="primary", use_container_width=True)

if predict_btn:
    if has_red:
        st.error(
            "Prediction blocked. One or more inputs fall outside the "
            "Q25-Q75 training data bounds. Please adjust the highlighted inputs."
        )
    else:
        x_SI = convert_display_to_si(vals, units)
        mr_mpa = compute_mr(model, x_SI)

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
                rows.append([
                    model, i + 1,
                    round(from_si_stress(theta_kPa, "US"), 3),
                    round(from_si_stress(tau_kPa, "US"), 3),
                    round(mpa_to_ksi(mr_i), 4),
                ])
            else:
                rows.append([
                    model, i + 1,
                    round(theta_kPa, 3),
                    round(tau_kPa, 3),
                    round(mr_i, 4),
                ])

        stress_u = "psi" if units == "US" else "kPa"
        mr_u = "ksi" if units == "US" else "MPa"

        df = pd.DataFrame(rows, columns=[
            "Model", "Seq",
            f"Bulk Stress ({stress_u})",
            f"Oct. Shear Stress ({stress_u})",
            f"MR ({mr_u})",
        ])

        # Sensitivity
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
                PARAM_SHORT[i],
                round(mr_base_d, 4),
                round(mrp_d, 4),
                round(mrm_d, 4),
                round(mrp_d - mr_base_d, 4),
                round(mrm_d - mr_base_d, 4),
            ])

        sens_df = pd.DataFrame(sens_rows, columns=[
            "Parameter",
            f"Baseline MR ({mr_u})",
            f"+10% MR ({mr_u})",
            f"-10% MR ({mr_u})",
            f"Delta+ ({mr_u})",
            f"Delta- ({mr_u})",
        ])

        st.session_state.pred_data = df
        st.session_state.sens_data = sens_df
        st.session_state.mr_value = mpa_to_ksi(mr_mpa) if units == "US" else mr_mpa
        st.session_state.mr_unit = mr_u
        st.session_state.last_s3 = s3_list
        st.session_state.last_sd = sd_list
        st.session_state.last_model = model
        st.session_state.last_units = units

# =====================================================================
#  RESULTS DISPLAY
# =====================================================================
if st.session_state.pred_data is not None and st.session_state.mr_value is not None:
    mr_u = st.session_state.mr_unit
    mr_v = st.session_state.mr_value

    st.markdown(
        f'<div class="mr-card">'
        f'<div class="mr-label">Predicted Resilient Modulus</div>'
        f'<div class="mr-value">{mr_v:.3f}</div>'
        f'<div class="mr-unit">{mr_u}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Chart ---
    st.markdown(
        '<div class="section-head">MR vs. Stress Sequence - AASHTO T 307</div>',
        unsafe_allow_html=True,
    )

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
        x_label = f"Bulk Stress ({stress_u})"
    else:
        x_data = df.iloc[:, 3]
        x_label = f"Octahedral Shear Stress ({stress_u})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_data, y=df[mr_col],
        mode="lines+markers+text",
        marker=dict(size=10, color="#2c5364", line=dict(width=1.5, color="white")),
        line=dict(width=2.5, color="#2c5364"),
        text=[str(i) for i in range(1, 16)],
        textposition="top center",
        textfont=dict(size=9, color="#999"),
        hovertemplate=f"{x_label}: %{{x:.2f}}<br>MR: %{{y:.3f}} {mr_u}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=f"Resilient Modulus ({mr_u})",
        title=dict(text=f"{st.session_state.last_model} - AASHTO T307 Sequences", font=dict(size=15)),
        template="plotly_white",
        height=430,
        margin=dict(l=55, r=25, t=50, b=45),
        font=dict(family="DM Sans, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    # --- Results Table ---
    st.markdown(
        '<div class="section-head">AASHTO T307 Prediction Results</div>',
        unsafe_allow_html=True,
    )

    lim = LIMITS_SI[st.session_state.last_model]
    tau_lo, tau_hi = lim[10, 0], lim[10, 1]
    theta_lo, theta_hi = lim[11, 0], lim[11, 1]

    def highlight_oor(row):
        idx = int(row["Seq"]) - 1
        theta_i = sd_list[idx] + 3 * s3_list[idx]
        tau_i = (math.sqrt(2) / 3) * sd_list[idx]
        if theta_i < theta_lo or theta_i > theta_hi or tau_i < tau_lo or tau_i > tau_hi:
            return ["background-color: #fde8e8"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(highlight_oor, axis=1).format(precision=3)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Rows highlighted in red: stress state outside Q25-Q75 training bounds (interpret with caution).")

    # --- Sensitivity ---
    st.markdown(
        '<div class="section-head">Sensitivity Analysis (+/-10% perturbation on each input)</div>',
        unsafe_allow_html=True,
    )
    sens_df = st.session_state.sens_data

    def color_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #27ae60; font-weight: 600"
            elif val < 0:
                return "color: #e74c3c; font-weight: 600"
        return ""

    delta_cols = [c for c in sens_df.columns if "Delta" in c]
    styled_sens = sens_df.style.map(color_delta, subset=delta_cols).format(precision=4)
    st.dataframe(styled_sens, use_container_width=True, hide_index=True)

    # --- Export ---
    st.markdown(
        '<div class="section-head">Export Results</div>',
        unsafe_allow_html=True,
    )
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_buf = io.StringIO()
        csv_buf.write("URMT Prediction Results\n")
        csv_buf.write(f"Model: {st.session_state.last_model}\n")
        csv_buf.write(f"Units: {st.session_state.last_units}\n\n")
        df.to_csv(csv_buf, index=False)
        csv_buf.write("\n\nSensitivity Analysis (+/-10%)\n")
        sens_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV",
            csv_buf.getvalue(),
            file_name="URMT_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        st.info("Tip: Use Ctrl+P (Cmd+P on Mac) to print/save this page as PDF.")

# =====================================================================
#  FOOTER
# =====================================================================
st.markdown(
    '<div class="footer">'
    "URMT v2.0 &mdash; Polynomial Ridge Regression (Deg. 3) &mdash; "
    "&copy; 2025 Michigan State University<br>"
    "Author: Mohamad Yaman Fares &middot; Civil &amp; Environmental Engineering &middot; faresmoh@msu.edu"
    "</div>",
    unsafe_allow_html=True,
)
