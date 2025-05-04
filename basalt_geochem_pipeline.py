# basalt_geochem_pipeline.py

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
    """Strict 45–57 wt% SiO₂ filter, only if SiO2 exists."""
    if 'SiO2' not in df.columns:
        return df
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute all custom geochemical ratios and ε-values."""
    eps = 1e-6
    # major/trace ratios
    df['Th/La']   = df['Th']  / (df['La']  + eps)
    df['Ce/Ce*']  = df['Ce']  / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']   = df['Nb']  / (df['Zr']  + eps)
    df['La/Zr']   = df['La']  / (df['Zr']  + eps)
    df['La/Yb']   = df['La']  / (df['Yb']  + eps)
    df['Sr/Y']    = df['Sr']  / (df['Y']   + eps)
    df['Gd/Yb']   = df['Gd']  / (df['Yb']  + eps)
    df['Nd/Nd*']  = df['Nd']  / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb']   = df['Dy']  / (df['Yb']  + eps)
    df['Th/Yb']   = df['Th']  / (df['Yb']  + eps)
    df['Nb/Yb']   = df['Nb']  / (df['Yb']  + eps)
    df['Zr/Y']    = df['Zr']  / (df['Y']   + eps)
    df['Ti/Zr']   = df['Ti']  / (df['Zr']  + eps)
    df['Y/Nb']    = df['Y']   / (df['Nb']  + eps)
    df['Th/Nb']   = df['Th']  / (df['Nb']  + eps)
    if 'Ti' in df.columns and 'V' in df.columns:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Ti' in df.columns and 'Al' in df.columns:
        df['Ti/Al']= df['Ti'] / (df['Al']+ eps)
    # isotope epsilons
    if '143Nd/144Nd' in df.columns:
        CHUR=0.512638
        df['εNd'] = ((df['143Nd/144Nd']-CHUR)/CHUR)*1e4
    if '176Hf/177Hf' in df.columns:
        CHUR=0.282785
        df['εHf'] = ((df['176Hf/177Hf']-CHUR)/CHUR)*1e4
    return df

def preprocess_list(df_list):
    """Concat, filter, and compute ratios in one go."""
    full = pd.concat(df_list, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")

st.markdown("""
Upload your own GEOROC CSV parts **or** load the prepared GEOROC training set from GitHub.
""")

use_github      = st.checkbox("📦 Load GEOROC parts from GitHub")
uploaded_parts  = st.file_uploader("Upload split GEOROC CSVs", type="csv", accept_multiple_files=True)

df_list, skipped = [], []

# — GitHub path —
if use_github:
    base_url = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    filenames = [f"2024-12-2JETOA_part{i}.csv" for i in range(1, 10)]
    for fn in filenames:
        try:
            r = requests.get(base_url + fn)
            part = load_data(StringIO(r.text))
            if part.isnull().any().any():
                skipped.append(fn)
                continue
            df_list.append(part)
        except Exception:
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
        except Exception:
            skipped.append(uf.name)
    if not df_list:
        st.error("❌ All uploaded files were invalid or contained missing data.")
        st.stop()
    df = preprocess_list(df_list)
    if skipped:
        st.warning("⚠ Skipped uploads: " + ", ".join(skipped))

# — Neither chosen —
else:
    st.info("▶️ Please select GitHub or upload CSV files to proceed.")
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
    ax2.set_title("REE Spider Plot (Chondrite-normalized)")
    st.pyplot(fig2)

st.subheader("Computed Ratios Summary")
ratio_cols = ['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb']
available  = [c for c in ratio_cols if c in df.columns]
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
    # auto-label any missing
    df.loc[df[label_col].isna(), 'Auto_Label'] = pd.Categorical.from_codes(
        mdl.predict(df[features]),
        categories=df[label_col].astype('category').cat.categories
    )

# -----------------------------
# Export
# -----------------------------
st.subheader("Download Results")
csv_bytes = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", data=csv_bytes, file_name="geochem_results.csv", mime="text/csv")
