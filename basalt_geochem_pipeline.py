# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# CONSTANTS & ELEMENT LISTS
# -----------------------------
CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 1.59,  'Pr': 0.456, 'Nd': 1.23,
    'Sm': 0.298, 'Eu': 0.123, 'Gd': 0.27,  'Tb': 0.1,
    'Dy': 0.26,  'Ho': 0.059, 'Er': 0.167, 'Tm': 0.025,
    'Yb': 0.17,  'Lu': 0.036
}
REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# HELPERS
# -----------------------------
def load_data(source):
    """
    Read a GEOROC CSV (from a file-like or URL text), rename
    common oxide/element headers to simple column names.
    """
    df = pd.read_csv(source)
    df.rename(columns={
        'SiO2 [wt%]': 'SiO2',
        'TiO2 [wt%]': 'Ti',
        'Al2O3 [wt%]': 'Al',
        'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO',
        'CaO [wt%]': 'CaO',
        'Na2O [wt%]': 'Na2O',
        'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5',
        'Th [ppm]': 'Th',
        'Nb [ppm]': 'Nb',
        'Zr [ppm]': 'Zr',
        'Y [ppm]': 'Y',
        'Sr [ppm]': 'Sr',
        'La [ppm]': 'La',
        'Ce [ppm]': 'Ce',
        'Pr [ppm]': 'Pr',
        'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm',
        'Eu [ppm]': 'Eu',
        'Gd [ppm]': 'Gd',
        'Tb [ppm]': 'Tb',
        'Dy [ppm]': 'Dy',
        'Ho [ppm]': 'Ho',
        'Er [ppm]': 'Er',
        'Tm [ppm]': 'Tm',
        'Yb [ppm]': 'Yb',
        'Lu [ppm]': 'Lu',
        'V [ppm]': 'V',
        'Sc [ppm]': 'Sc',
        'Co [ppm]': 'Co',
        'Ni [ppm]': 'Ni',
        'Cr [ppm]': 'Cr',
        'Hf [ppm]': 'Hf',
    }, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only 45–57 wt% SiO₂ entries, but skip if SiO2 is missing."""
    if 'SiO2' not in df.columns:
        return df
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """
    Compute trace/major ratios and ε‐isotopes, only if input
    columns exist. Returns the same df with new columns appended.
    """
    eps = 1e-6

    def add_ratio(name, num, den):
        if num in df.columns and den in df.columns:
            df[name] = df[num] / (df[den] + eps)

    # Basic ratios
    add_ratio('Th/La',   'Th', 'La')
    if all(x in df.columns for x in ['Ce','La','Pr']):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    add_ratio('Nb/Zr',   'Nb', 'Zr')
    add_ratio('La/Zr',   'La', 'Zr')
    add_ratio('La/Yb',   'La', 'Yb')
    add_ratio('Sr/Y',    'Sr', 'Y')
    add_ratio('Gd/Yb',   'Gd', 'Yb')
    if all(x in df.columns for x in ['Nd','Pr','Sm']):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    add_ratio('Dy/Yb',   'Dy', 'Yb')
    add_ratio('Th/Yb',   'Th', 'Yb')
    add_ratio('Nb/Yb',   'Nb', 'Yb')
    add_ratio('Zr/Y',    'Zr', 'Y')
    add_ratio('Ti/Zr',   'Ti', 'Zr')
    add_ratio('Y/Nb',    'Y',  'Nb')
    add_ratio('Th/Nb',   'Th', 'Nb')
    add_ratio('Ti/V',    'Ti', 'V')
    add_ratio('Ti/Al',   'Ti', 'Al')

    # Epsilon isotope values
    if '143Nd/144Nd' in df.columns:
        CHUR = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR) / CHUR) * 1e4
    if '176Hf/177Hf' in df.columns:
        CHUR = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR) / CHUR) * 1e4

    return df

def preprocess_list(parts):
    """Concat uploaded/remote parts, filter, compute ratios."""
    full = pd.concat(parts, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------
st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")
st.markdown("**Load** GEOROC parts from GitHub **or** **Upload** your own CSVs.")

use_github     = st.checkbox("📦 Load GEOROC parts from GitHub")
uploaded_parts = st.file_uploader("Upload split GEOROC CSVs", type="csv", accept_multiple_files=True)

df_parts, skipped = [], []

# — GitHub path —
if use_github:
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    # Adjust these names to match your repo
    names = [f"2024-12-2JETOA_part{i}.csv" for i in range(1, 11)]
    for fn in names:
        try:
            r = requests.get(base + fn)
            part = load_data(StringIO(r.text))
            if part.isnull().any().any():
                skipped.append(fn)
                continue
            df_parts.append(part)
        except:
            skipped.append(fn)
    if not df_parts:
        st.error("❌ No valid GEOROC parts could be loaded.")
        st.stop()
    df = preprocess_list(df_parts)
    if skipped:
        st.warning("⚠ Skipped: " + ", ".join(skipped))

# — Upload path —
elif uploaded_parts:
    for up in uploaded_parts:
        try:
            part = load_data(up)
            if part.isnull().any().any():
                skipped.append(up.name)
                continue
            df_parts.append(part)
        except:
            skipped.append(up.name)
    if not df_parts:
        st.error("❌ All uploaded files were invalid or incomplete.")
        st.stop()
    df = preprocess_list(df_parts)
    if skipped:
        st.warning("⚠ Skipped uploads: " + ", ".join(skipped))

# — Neither chosen —
else:
    st.info("▶️ Select GitHub **or** upload files to begin.")
    st.stop()

# -----------------------------
# PLOTS & SUMMARY
# -----------------------------
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set(xlabel="εNd", ylabel="εHf", title="εNd vs εHf")
    st.pyplot(fig)

if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, r in df.iterrows():
        norm = [r[e] / CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, norm, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set(title="REE Spider Plot")
    st.pyplot(fig2)

st.subheader("Computed Ratios Summary")
ratio_cols = ['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb']
avail = [c for c in ratio_cols if c in df.columns]
if avail:
    st.dataframe(df[avail].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# ML CLASSIFICATION
# -----------------------------
st.subheader("Machine-Learning Classification (Tectonic Setting)")
labels  = [c for c in df.columns if df[c].nunique() < 20]
features = df.select_dtypes("number").columns.tolist()

label_col = st.selectbox("Label column", labels)
sel_feats = st.multiselect("Feature columns", features, default=avail[:5])

if st.button("Run Classification") and sel_feats:
    dfc = df.dropna(subset=sel_feats + [label_col])
    X = dfc[sel_feats]
    y = dfc[label_col].astype('category').cat.codes
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    mdl = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, early_stopping=True)
    mdl.fit(Xtr, ytr)
    preds = mdl.predict(Xte)
    st.text(classification_report(yte, preds, zero_division=0))

# -----------------------------
# DOWNLOAD
# -----------------------------
st.subheader("Download Processed Results")
csv_bytes = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", data=csv_bytes, file_name="geochem_results.csv", mime="text/csv")
