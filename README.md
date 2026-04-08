# URMT - The Ultimate Resilient Modulus Tool (Web Version)

**Streamlit web application** for predicting the Resilient Modulus (MR) of pavement materials using Polynomial Ridge Regression (Degree 3).

Ported from the MATLAB "Prediction Tool" tab of the original URMT by **Mohamad Yaman Fares**, Civil & Environmental Engineering, Michigan State University.

---

## Features

- **Two material models**: Base/Subbase and Subgrade
- **12 soil index property inputs** with live validation
- **Unit toggle**: SI (mm, kN/m³, kPa, MPa) ↔ US (in, pcf, psi, ksi)
- **Q25–Q75 IQR validation** with color-coded status lamps (green / yellow / red)
- **Gravel + Sand + Fines = 100%** live sum check
- **D10 ≤ D30 ≤ D60** grain-size ordering check
- **AASHTO T 307-99** stress sequence prediction (15 sequences)
- **Interactive Plotly chart** with selectable X-axis
- **Sensitivity analysis** (±10% perturbation on each input)
- **CSV export** of results and sensitivity data
- **Load Median Defaults** for quick testing

---

## Usage

1. Select **Unit System** (SI or US) and **Material Model** in the sidebar
2. Enter 12 soil properties (or click **Load Median Defaults**)
3. Check that all status lamps are green/yellow (red = out of training range)
4. Click **Predict Resilient Modulus**
5. Review the MR value, AASHTO T307 table, chart, and sensitivity analysis
6. Export results via **Download CSV**

---

© 2025 Michigan State University — Mohamad Yaman Fares
