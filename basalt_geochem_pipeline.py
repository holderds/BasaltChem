# basalt_geochem_pipeline.py

import os
import glob
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Configuration
# -----------------------------
EPS = 1e-6

CHONDRITE_VALUES = {
    'La': 0.237, 'Ce': 1.59, 'Pr': 0.327, 'Nd': 1.23, 'Sm': 0.303,
    'Eu': 0.094, 'Gd': 0.27,  'Tb': 0.044, 'Dy': 0.26, 'Ho': 0.056,
    'Er': 0.165, 'Tm': 0.025, 'Yb': 0.17,  'Lu': 0.025
}
REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# Function Definitions
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read CSV, coerce to numeric, rename key geochem columns."""
    df = pd.read_csv(path, low_memory=False)
    # force anything that looks numeric into numeric dtype
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # rename common oxide & trace-element columns
    rename_map = {
        'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti',    'Al2O3 [wt%]': 'Al',
        'Fe2O3(t) [wt%]': 'Fe2O3','MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
        'Na2O [wt%]': 'Na2O',    'K2O [wt%]': 'K2O',  'P2O5 [wt%]': 'P2O5',
        'Th [ppm]': 'Th',        'Nb [ppm]': 'Nb',    'Zr [ppm]': 'Zr',
        'Y [ppm]': 'Y',          'Sr [ppm]': 'Sr',    'La [ppm]': 'La',
        'Ce [ppm]': 'Ce',        'Pr [ppm]': 'Pr',    'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm',        'Eu [ppm]': 'Eu',    'Gd [ppm]': 'Gd',
        'Tb [ppm]': 'Tb',        'Dy [ppm]': 'Dy',    'Ho [ppm]': 'Ho',
        'Er [ppm]': 'Er',        'Tm [ppm]': 'Tm',    'Yb [ppm]': 'Yb',
        'Lu [ppm]': 'Lu',        'V [ppm]': 'V',      'Sc [ppm]': 'Sc',
        'Co [ppm]': 'Co',        'Ni [ppm]': 'Ni',    'Cr [ppm]': 'Cr',
        'Hf [ppm]': 'Hf'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def filter_basaltic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only samples with 45–57 wt% SiO2."""
    if 'SiO2' not in df.columns:
        st.error("❌ ‘SiO2’ column missing—check your data.")
        st.stop()
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)].copy()

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all custom major/trace ratios and REE‐normalized indices."""
    out = df.copy()
    # Major/trace ratios
    out['Th/La']  = out['Th']   / (out['La']   + EPS)
    out['Nb/Zr']  = out['Nb']   / (out['Zr']   + EPS)
    out['La/Zr']  = out['La']   / (out['Zr']   + EPS)
    out['La/Yb']  = out['La']   / (out['Yb']   + EPS)
    out['Gd/Yb']  = out['Gd']   / (out['Yb']   + EPS)
    out['Dy/Yb']  = out['Dy']   / (out['Yb']   + EPS)
    if 'Sr' in out.columns and 'Y' in out.columns:
        out['Sr/Y'] = out['Sr'] / (out['Y'] + EPS)
    out['Th/Nb']  = out['Th']   / (out['Nb']   + EPS)
    # Multi-element anomaly indices
    out['Ce/Ce*'] = out['Ce']   / (np.sqrt(out['La'] * out['Pr']) + EPS)
    out['Eu/Eu*'] = out['Eu']   / (np.sqrt(out['Sm'] * out['Gd']) + EPS)
    out['Nb/Nb*'] = out['Nb']   / (np.sqrt(out['Zr'] * out['Y'])  + EPS)
    return out

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("Basalt & Basaltic‐Andesite Geochemistry Interpreter")

# 1️⃣ Auto‐load all CSVs matching prefix
csv_paths = glob.glob("2024-12-2JETOA_*.csv")
if not csv_paths:
    st.error("❌ No training CSVs found (expected 2024-12-2JETOA_*.csv).")
    st.stop()

parts, skipped = [], []
for fp in csv_paths:
    try:
        dfp = load_data(fp)
        # require at least one valid SiO2 value
        if dfp['SiO2'].notna().any():
            parts.append(dfp)
        else:
            skipped.append(os.path.basename(fp))
    except Exception:
        skipped.append(os.path.basename(fp))

if not parts:
    st.error("❌ Could not load any valid training CSVs.")
    st.stop()
if skipped:
    st.warning(f"⚠ Skipped {len(skipped)} file(s): {', '.join(skipped)}")

# 2️⃣ Concatenate → filter → compute
df = pd.concat(parts, ignore_index=True)
df = filter_basaltic(df)
df = compute_ratios(df)

# -----------------------------
# REE Spider Plot
# -----------------------------
if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e] / CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, normed, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_title("REE Spider Diagram (Chondrite‐normalized)")
    ax2.set_ylabel("X/X<sub>chondrite</sub> (log scale)")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# -----------------------------
# Computed Ratios Summary
# -----------------------------
ratio_cols = [c for c in [
    'Th/La','Nb/Zr','La/Zr','La/Yb','Gd/Yb','Dy/Yb',
    'Ce/Ce*','Eu/Eu*','Nb/Nb*','Sr/Y'
] if c in df.columns]

if ratio_cols:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))
else:
    st.info("ℹ️ No ratios computed—check for required elements in your data.")

# -----------------------------
# Download Cleaned Data & Last Plot
# -----------------------------
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Cleaned Data (CSV)", csv_bytes, "processed_geochem.csv", "text/csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Download Last Plot (PNG)", buf.getvalue(), "plot.png", "image/png")

# -----------------------------
# Supervised Classification
# -----------------------------
st.subheader("Supervised Classification (Tectonic Setting)")
label_opts = [c for c in df.columns if df[c].nunique() < 20]
label_col = st.selectbox("Select label column", label_opts)

num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
default_feats = num_feats[:8]
feats = st.multiselect("Select features", num_feats, default=default_feats)

if label_col and feats:
    subset = df.dropna(subset=feats + [label_col])
    X, y = subset[feats], subset[label_col].astype('category').cat.codes

    if len(X) < 2:
        st.warning("Not enough data (need ≥2 samples) for classification.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        clf = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

