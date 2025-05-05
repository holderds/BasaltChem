# basalt_geochem_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Constants
# -----------------------------
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al',
    'Fe2O3(t) [wt%]': 'Fe2O3', 'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
    'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O', 'P2O5 [wt%]': 'P2O5',
    'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr', 'Y [ppm]': 'Y',
    'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb',
    'Dy [ppm]': 'Dy', 'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm',
    'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu', 'V [ppm]': 'V', 'Sc [ppm]': 'Sc',
    'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf',
    'Sr [ppm]': 'Sr', '143Nd/144Nd': '143Nd/144Nd', '176Hf/177Hf': '176Hf/177Hf',
    '87Sr/86Sr': '87Sr/86Sr', '206Pb/204Pb': '206Pb/204Pb', '207Pb/204Pb': '207Pb/204Pb'
}

GITHUB_CSV_URLS = [
    f"https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part{i}.csv"
    for i in range(1,6)
]

CHONDRITE_VALUES = {
    'La': 0.237, 'Ce': 1.59, 'Pr': 0.54, 'Nd': 1.23, 'Sm': 0.31,
    'Eu': 0.088, 'Gd': 0.27, 'Tb': 0.036, 'Dy': 0.26, 'Ho': 0.056,
    'Er': 0.165, 'Tm': 0.025, 'Yb': 0.17, 'Lu': 0.025
}

REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# Helper Functions
# -----------------------------

def load_and_rename(csv_source):
    df = pd.read_csv(csv_source)
    df.rename(columns=RENAME_MAP, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only 45–57 wt% SiO2; drop rows missing SiO2."""
    if 'SiO2' not in df.columns:
        return df.iloc[0:0]
    return df.dropna(subset=['SiO2']).query("SiO2 >= 45 and SiO2 <= 57")

def compute_ratios(df):
    """Compute a suite of ratios, only if raw columns exist."""
    eps = 1e-6
    # Major & trace ratios
    if {'Th','La'}.issubset(df.columns):
        df['Th/La'] = df['Th'] / (df['La'] + eps)
    if {'Ce','La','Pr'}.issubset(df.columns):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    if {'Nb','Zr'}.issubset(df.columns):
        df['Nb/Zr'] = df['Nb'] / (df['Zr'] + eps)
        df['La/Zr'] = df['La'] / (df['Zr'] + eps) if 'La' in df.columns else np.nan
    if {'La','Yb'}.issubset(df.columns):
        df['La/Yb'] = df['La'] / (df['Yb'] + eps)
    if {'Sr','Y'}.issubset(df.columns):
        df['Sr/Y'] = df['Sr'] / (df['Y'] + eps)
    if {'Gd','Yb'}.issubset(df.columns):
        df['Gd/Yb'] = df['Gd'] / (df['Yb'] + eps)
    if {'Nd','Pr','Sm'}.issubset(df.columns):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    if {'Dy','Yb'}.issubset(df.columns):
        df['Dy/Yb'] = df['Dy'] / (df['Yb'] + eps)
    if {'Th','Nb'}.issubset(df.columns):
        df['Th/Nb'] = df['Th'] / (df['Nb'] + eps)
    if {'Nb','Yb'}.issubset(df.columns):
        df['Nb/Yb'] = df['Nb'] / (df['Yb'] + eps)
    if {'Ti','Zr'}.issubset(df.columns):
        df['Ti/Zr'] = df['Ti'] / (df['Zr'] + eps)
    if {'Y','Nb'}.issubset(df.columns):
        df['Y/Nb'] = df['Y'] / (df['Nb'] + eps)
    if {'Ti','V'}.issubset(df.columns):
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if {'Ti','Al'}.issubset(df.columns):
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)

    # Isotopic ε‐values
    if '143Nd/144Nd' in df.columns:
        CHUR_Nd = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR_Nd) / CHUR_Nd) * 1e4
    if '176Hf/177Hf' in df.columns:
        CHUR_Hf = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR_Hf) / CHUR_Hf) * 1e4
    if {'206Pb/204Pb','207Pb/204Pb'}.issubset(df.columns):
        df['Pb_iso_ratio'] = df['206Pb/204Pb'] / df['207Pb/204Pb']
    return df

def preprocess_list(df_list):
    """Concatenate, filter silica, compute ratios."""
    full = pd.concat(df_list, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("""
Upload your split GEOROC CSVs or load the cleaned example dataset from GitHub.
""")
use_github = st.checkbox("Load example GEOROC parts from GitHub")
uploads    = st.file_uploader("Upload split GEOROC .csv files", type="csv", accept_multiple_files=True)

df_list = []
skipped  = []

if use_github:
    # load from GitHub URLs
    for url in GITHUB_CSV_URLS:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df_part = pd.read_csv(StringIO(r.text))
            df_part.rename(columns=RENAME_MAP, inplace=True)
            df_list.append(df_part)
        except Exception:
            skipped.append(url.split("/")[-1])
elif uploads:
    # load from local uploads
    for up in uploads:
        try:
            df_part = load_and_rename(up)
            df_list.append(df_part)
        except Exception:
            skipped.append(up.name)
else:
    st.warning("Please either upload CSVs or select the GitHub example.")
    st.stop()

if not df_list:
    st.error("❌ Could not load any valid GEOROC parts.")
    if skipped:
        st.info("Attempted but failed to load:\n  " + "\n  ".join(skipped))
    st.stop()

# combine + preprocess
df = preprocess_list(df_list)

if skipped:
    st.warning(f"⚠️ Skipped files: {', '.join(skipped)}")

if df.empty:
    st.error("❌ No data after filtering SiO₂ = 45–57 wt%.")
    st.stop()

# -----------------------------
# εNd vs εHf Plot
if {'εNd','εHf'}.issubset(df.columns):
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd")
    ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state['last_fig'] = fig

# -----------------------------
# REE Spider Plot
if all(elem in df.columns for elem in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, r in df.iterrows():
        normed = [r[elem]/CHONDRITE_VALUES[elem] for elem in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, normed, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set_ylabel("Normalized to Chondrite")
    ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# -----------------------------
# Preview Computed Ratios
ratio_cols = [c for c in ['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb',
                          'Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Nb','Nb/Yb']
              if c in df.columns]
if ratio_cols:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))
else:
    st.warning("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# Download & Export
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download All Data as CSV", csv_data, "georoc_classified.csv", "text/csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Download Last Figure", buf.getvalue(), "figure.png", "image/png")

# -----------------------------
# Simple MLP Classification Example
st.subheader("MLP Classification (Tectonic Setting)")
label_options = [c for c in df.columns if df[c].dtype in [np.int64, np.float64, object] and df[c].nunique()<20]
label_col = st.selectbox("Label Column", label_options) if label_options else None
feat_cols = st.multiselect("Feature Columns", df.select_dtypes("number").columns.tolist())

if label_col and feat_cols:
    df_ml = df.dropna(subset=feat_cols + [label_col])
    X = df_ml[feat_cols]
    y = df_ml[label_col].astype('category').cat.codes
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    rep = classification_report(yte, preds, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).transpose())
