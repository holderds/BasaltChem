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
    'Eu': 0.094, 'Gd': 0.27, 'Tb': 0.044, 'Dy': 0.26, 'Ho': 0.056,
    'Er': 0.165, 'Tm': 0.025, 'Yb': 0.17, 'Lu': 0.025
}

REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# Function Definitions
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read CSV, coerce numerics, rename geochem columns."""
    df = pd.read_csv(path, low_memory=False)
    # convert any column that looks numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # rename the common oxide & trace names
    rename_map = {
        'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al',
        'Fe2O3(t) [wt%]': 'Fe2O3', 'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
        'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O', 'P2O5 [wt%]': 'P2O5',
        'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr', 'Y [ppm]': 'Y',
        'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb',
        'Dy [ppm]': 'Dy', 'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm',
        'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu', 'V [ppm]': 'V', 'Sc [ppm]': 'Sc',
        'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def filter_basaltic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 45–57 wt% SiO2."""
    if 'SiO2' not in df.columns:
        st.error("SiO2 column missing—check your data.")
        st.stop()
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute custom ratios (including Nb/Nb*, Ce/Ce*, Eu/Eu*)."""
    df = df.copy()
    # simple major/trace ratios
    df['Th/La']   = df['Th']   / (df['La']  + EPS)
    df['Nb/Zr']   = df['Nb']   / (df['Zr']  + EPS)
    df['La/Zr']   = df['La']   / (df['Zr']  + EPS)
    df['La/Yb']   = df['La']   / (df['Yb']  + EPS)
    df['Sr/Y']    = df['Sr']   / (df['Y']   + EPS) if 'Sr' in df.columns else np.nan
    df['Gd/Yb']   = df['Gd']   / (df['Yb']  + EPS)
    df['Dy/Yb']   = df['Dy']   / (df['Yb']  + EPS)
    df['Th/Nb']   = df['Th']   / (df['Nb']  + EPS)
    # multi-element indices
    df['Ce/Ce*']  = df['Ce']   / (np.sqrt(df['La'] * df['Pr']) + EPS)
    df['Eu/Eu*']  = df['Eu']   / (np.sqrt(df['Sm'] * df['Gd']) + EPS)
    df['Nb/Nb*']  = df['Nb']   / (np.sqrt(df['Zr'] * df['Y'])  + EPS)
    return df

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")

# 1️⃣ Load all training CSVs by prefix
csv_paths = glob.glob("2024-12-2JETOA_*.csv")
if not csv_paths:
    st.error("❌ No training CSVs found—make sure files are named 2024-12-2JETOA_*.csv in the repo root.")
    st.stop()

df_parts, skipped = [], []
for p in csv_paths:
    try:
        part = load_data(p)
        # drop rows that utterly lack SiO2
        if part['SiO2'].notna().any():
            df_parts.append(part)
        else:
            skipped.append(os.path.basename(p))
    except Exception:
        skipped.append(os.path.basename(p))

if not df_parts:
    st.error("❌ Could not load any training CSVs.")
    st.stop()
if skipped:
    st.warning(f"⚠ Skipped {len(skipped)} file(s): {', '.join(skipped)}")

# 2️⃣ Concat → Filter → Compute
df = pd.concat(df_parts, ignore_index=True)
df = filter_basaltic(df)
df = compute_ratios(df)

# -----------------------------
# εNd vs εHf Plot (if available)
# -----------------------------
if {'εNd','εHf'}.issubset(df.columns):
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state['last_fig'] = fig

# -----------------------------
# REE Spider Plot
# -----------------------------
if all(elem in df.columns for elem in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e]/CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, normed, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set_title("REE Spider Diagram (Normalized)")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# -----------------------------
# Computed Ratios Summary
# -----------------------------
ratio_cols = [c for c in ['Th/La','Nb/Zr','La/Zr','La/Yb','Gd/Yb','Dy/Yb','Ce/Ce*','Eu/Eu*','Nb/Nb*'] if c in df.columns]
if ratio_cols:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# Download Buttons
# -----------------------------
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Processed Data as CSV", csv_bytes, "processed_geochem.csv", "text/csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Download Last Plot as PNG", buf.getvalue(), "last_plot.png", "image/png")

# -----------------------------
# Supervised Classification
# -----------------------------
st.subheader("Supervised Classification (Tectonic Setting)")
# choose a label column with <20 unique categories
labels = [c for c in df.columns if df[c].nunique() < 20]
label = st.selectbox("Label column", labels)

# numeric features
num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
default_feats = num_feats[:8]
feats = st.multiselect("Features for classification", num_feats, default=default_feats)

if label and feats:
    # drop any rows missing either features or label
    cls = df.dropna(subset=feats+[label])
    X, y = cls[feats], cls[label].astype('category').cat.codes
    if len(X) < 2:
        st.warning("Not enough data for training—need at least 2 samples.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
