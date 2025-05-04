## basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Constants & Element Lists
# -----------------------------
CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 1.59, 'Pr': 0.456, 'Nd': 1.23,
    'Sm': 0.298, 'Eu': 0.123, 'Gd': 0.27, 'Tb': 0.1,
    'Dy': 0.26, 'Ho': 0.059, 'Er': 0.167, 'Tm': 0.025,
    'Yb': 0.17, 'Lu': 0.036
}
REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# Helper Functions
# -----------------------------
def load_data(source):
    """Read & rename GEOROC CSV columns to standard names."""
    df = pd.read_csv(source)
    df.rename(columns={
        'SiO2 [wt%]':'SiO2','TiO2 [wt%]':'Ti','Al2O3 [wt%]':'Al',
        'Fe2O3(t) [wt%]':'Fe2O3','MgO [wt%]':'MgO','CaO [wt%]':'CaO',
        'Na2O [wt%]':'Na2O','K2O [wt%]':'K2O','P2O5 [wt%]':'P2O5',
        'Th [ppm]':'Th','Nb [ppm]':'Nb','Zr [ppm]':'Zr','Y [ppm]':'Y',
        'La [ppm]':'La','Ce [ppm]':'Ce','Pr [ppm]':'Pr','Nd [ppm]':'Nd',
        'Sm [ppm]':'Sm','Eu [ppm]':'Eu','Gd [ppm]':'Gd','Tb [ppm]':'Tb',
        'Dy [ppm]':'Dy','Ho [ppm]':'Ho','Er [ppm]':'Er','Tm [ppm]':'Tm',
        'Yb [ppm]':'Yb','Lu [ppm]':'Lu','V [ppm]':'V','Sc [ppm]':'Sc',
        'Co [ppm]':'Co','Ni [ppm]':'Ni','Cr [ppm]':'Cr','Hf [ppm]':'Hf',
        'Sr [ppm]':'Sr'
    }, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Strict 45–57 wt% SiO₂ filter, only if SiO₂ exists."""
    if 'SiO2' not in df.columns:
        return df
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute all custom geochemical ratios and ε-values, only if inputs exist."""
    eps = 1e-6

    # Helper to add ratio only if both cols exist
    def add_ratio(out, num, den):
        if num in df.columns and den in df.columns:
            df[out] = df[num] / (df[den] + eps)

    # Major/trace ratios
    add_ratio('Th/La', 'Th', 'La')
    if all(c in df.columns for c in ['Ce','La','Pr']):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    add_ratio('Nb/Zr', 'Nb', 'Zr')
    add_ratio('La/Zr', 'La', 'Zr')
    add_ratio('La/Yb', 'La', 'Yb')
    add_ratio('Sr/Y', 'Sr', 'Y')
    add_ratio('Gd/Yb', 'Gd', 'Yb')
    if all(c in df.columns for c in ['Nd','Pr','Sm']):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    add_ratio('Dy/Yb', 'Dy', 'Yb')
    add_ratio('Th/Yb', 'Th', 'Yb')
    add_ratio('Nb/Yb', 'Nb', 'Yb')
    add_ratio('Zr/Y', 'Zr', 'Y')
    add_ratio('Ti/Zr', 'Ti', 'Zr')
    add_ratio('Y/Nb', 'Y', 'Nb')
    add_ratio('Th/Nb', 'Th', 'Nb')
    add_ratio('Ti/V', 'Ti', 'V')
    add_ratio('Ti/Al', 'Ti', 'Al')

    # Isotope epsilon values
    if '143Nd/144Nd' in df.columns:
        CHUR_Nd = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR_Nd) / CHUR_Nd) * 1e4
    if '176Hf/177Hf' in df.columns:
        CHUR_Hf = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR_Hf) / CHUR_Hf) * 1e4

    return df

def preprocess_list(df_list):
    """Concat, filter, and compute ratios in one go."""
    full = pd.concat(df_list, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")

st.markdown("**Load** GEOROC parts from GitHub **or** **Upload** your own CSVs.")
use_github     = st.checkbox("📦 Load GEOROC parts from GitHub")
uploaded_parts = st.file_uploader("Upload split GEOROC CSVs", type="csv", accept_multiple_files=True)

df_list, skipped = [], []

# — GitHub path —
if use_github:
    base_url = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    filenames = [f"2024-12-2JETOA_part{i}.csv" for i in range(1, 11)]
    for fn in filenames:
        try:
            resp = requests.get(base_url + fn)
            part = load_data(StringIO(resp.text))
            if part.isnull().any().any():
                skipped.append(fn)
                continue
            df_list.append(part)
        except:
            skipped.append(fn)
    if not df_list:
        st.error("❌ No valid GEOROC parts loaded from GitHub.")
        st.stop()
    df = preprocess_list(df_list)
    if skipped:
        st.warning("⚠ Skipped: " + ", ".join(skipped))

# — Upload path —
elif uploaded_parts:
    for uf in uploaded_parts:
        try:
            part = load_data(uf)
            if part.isnull().any().any():
                skipped.append(uf.name)
                continue
            df_list.append(part)
        except:
            skipped.append(uf.name)
    if not df_list:
        st.error("❌ All uploaded files were invalid or contained missing data.")
        st.stop()
    df = preprocess_list(df_list)
    if skipped:
        st.warning("⚠ Skipped uploads: " + ", ".join(skipped))

# — Neither chosen —
else:
    st.info("▶️ Please select GitHub or upload files to proceed.")
    st.stop()

# -----------------------------
# Plots & Summaries
# -----------------------------
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set(xlabel="εNd", ylabel="εHf", title="εNd vs εHf")
    st.pyplot(fig)

if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        norm = [row[e] / CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, norm, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)

st.subheader("Computed Ratios Summary")
ratio_cols = ['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb']
available = [c for c in ratio_cols if c in df.columns]
st.dataframe(df[available].describe().T.style.format("{:.2f}"))

# -----------------------------
# Machine Learning Classification
# -----------------------------
st.subheader("ML Classification: Tectonic Setting")
label_col = st.selectbox("Label column", [c for c in df.columns if df[c].nunique() < 20])
features  = st.multiselect("Features", df.select_dtypes("number").columns.tolist(), default=available)

if st.button("Run Classification") and features:
    dfc = df.dropna(subset=features + [label_col])
    X, y = dfc[features], dfc[label_col].astype('category').cat.codes
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    mdl = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, early_stopping=True)
    mdl.fit(Xtr, ytr)
    preds = mdl.predict(Xte)
    st.text(classification_report(yte, preds, zero_division=0))

    # Auto-label missing
    if label_col in df.columns:
        missing = df[label_col].isna()
        df.loc[missing, 'Auto_Label'] = pd.Categorical.from_codes(
            mdl.predict(df.loc[missing, features]),
            categories=df[label_col].astype('category').cat.categories
        )

# -----------------------------
# Export
# -----------------------------
st.subheader("Download Results")
csv_bytes = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", data=csv_bytes, file_name="geochem_results.csv", mime="text/csv")

