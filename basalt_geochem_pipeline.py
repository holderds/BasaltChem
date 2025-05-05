# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

# -----------------------------
# Column rename map
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
    'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
}

# -----------------------------
# Helpers
# -----------------------------
def load_and_clean(csv_source):
    """Read CSV, rename columns, coerce all to numeric (NaN on failure)."""
    df = pd.read_csv(csv_source, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce all columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def get_github_csv_urls():
    """List all files in the main repo and return raw URLs for those
       starting with the given prefix."""
    api = "https://api.github.com/repos/holderds/BasaltChem/contents"
    resp = requests.get(api)
    resp.raise_for_status()
    files = resp.json()
    urls = []
    prefix = "2024-12-2JETOA_"
    for item in files:
        name = item.get("name","")
        if name.startswith(prefix) and name.lower().endswith(".csv"):
            raw = f"https://raw.githubusercontent.com/holderds/BasaltChem/main/{name}"
            urls.append(raw)
    return urls

def load_remote_parts():
    urls = get_github_csv_urls()
    df_list, skipped = [], []
    for url in urls:
        try:
            txt = requests.get(url).text
            dfp = load_and_clean(StringIO(txt))
            if dfp.empty or 'SiO2' not in dfp.columns:
                skipped.append(url.split("/")[-1])
                continue
            df_list.append(dfp)
        except Exception:
            skipped.append(url.split("/")[-1])
    return df_list, skipped

def load_uploaded_parts(uploaded_files):
    df_list, skipped = [], []
    for uf in uploaded_files:
        try:
            dfp = load_and_clean(uf)
            if dfp.empty or 'SiO2' not in dfp.columns:
                skipped.append(uf.name)
                continue
            df_list.append(dfp)
        except Exception:
            skipped.append(uf.name)
    return df_list, skipped

def compute_ratios(df):
    eps = 1e-6
    # Classic REE/major ratios if data present
    if {'Ce','La','Pr'}.issubset(df.columns):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    if {'Eu','Sm','Gd'}.issubset(df.columns):
        df['Eu/Eu*'] = df['Eu'] / (np.sqrt(df['Sm'] * df['Gd']) + eps)
    if {'Nb','Zr','Y'}.issubset(df.columns):
        df['Nb/Nb*'] = df['Nb'] / (np.sqrt(df['Zr'] * df['Y']) + eps)
    # Some simple major/trace ratios
    pairs = [
        ('Th','La','Th/La'), ('Nb','Zr','Nb/Zr'),
        ('La','Zr','La/Zr'), ('La','Yb','La/Yb'),
        ('Sr','Y','Sr/Y'), ('Gd','Yb','Gd/Yb'),
        ('Dy','Yb','Dy/Yb'), ('Th','Nb','Th/Nb'),
        ('Nb','Yb','Nb/Yb')
    ]
    for num, den, name in pairs:
        if num in df.columns and den in df.columns:
            df[name] = df[num] / (df[den] + eps)
    return df

# -----------------------------
# Load data (remote vs. upload)
# -----------------------------
use_remote = st.checkbox("Load training data from GitHub CSVs")
if use_remote:
    df_parts, skipped = load_remote_parts()
    if not df_parts:
        st.error("❌ Could not load any valid GEOROC parts.")
        if skipped:
            st.write("Failed to load:", ", ".join(skipped))
        st.stop()
    df = pd.concat(df_parts, ignore_index=True)
    if skipped:
        st.warning(f"Skipped {len(skipped)} parts: {', '.join(skipped)}")
else:
    uploaded = st.file_uploader("Upload GEOROC CSV parts", type="csv",
                                accept_multiple_files=True)
    if not uploaded:
        st.warning("Please upload at least one CSV part or load from GitHub.")
        st.stop()
    df_parts, skipped = load_uploaded_parts(uploaded)
    if not df_parts:
        st.error("❌ Could not load any valid uploaded parts.")
        if skipped:
            st.write("Failed to load:", ", ".join(skipped))
        st.stop()
    df = pd.concat(df_parts, ignore_index=True)
    if skipped:
        st.warning(f"Skipped {len(skipped)} uploads: {', '.join(skipped)}")

# -----------------------------
# Filter basaltic (45–57 SiO2)
# -----------------------------
if 'SiO2' not in df.columns:
    st.error("🛑 SiO₂ column missing—check your data.")
    st.stop()

df = df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]
st.write(f"Loaded {len(df)} samples (45–57 wt% SiO₂)")

# -----------------------------
# Compute ratios & anomalies
# -----------------------------
df = compute_ratios(df)

# -----------------------------
# Quick preview of key ratios
# -----------------------------
ratios = [c for c in ['Ce/Ce*','Eu/Eu*','Nb/Nb*','Th/La','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb'] if c in df.columns]
if ratios:
    st.subheader("Key Ratios Snapshot")
    st.dataframe(df[ratios].describe().T.style.format("{:.2f}"))

# -----------------------------
# Classification
# -----------------------------
st.subheader("Supervised Classification: Tectonic Setting")
numeric = df.select_dtypes(include="number").columns.tolist()
label = st.selectbox("Label column", [c for c in df.columns if df[c].nunique()<20])
features = st.multiselect("Features", numeric, default=numeric[:10])

if label and features:
    cls_df = df.dropna(subset=features + [label])
    if len(cls_df) < 2:
        st.warning("Not enough data for classification.")
    else:
        X = cls_df[features]
        y = cls_df[label].astype("category").cat.codes
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                           test_size=0.3,
                                                           random_state=42)
        model = MLPClassifier(hidden_layer_sizes=(50,25),
                              max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        st.subheader("Classification Report")
        st.text(report)

        # Download
        if st.button("Download labeled CSV"):
            out = cls_df.copy()
            out["Predicted"] = model.predict(df[features].fillna(0))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv,
                               file_name="classified_geochem.csv",
                               mime="text/csv")


