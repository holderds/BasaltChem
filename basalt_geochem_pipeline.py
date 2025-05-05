# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import streamlit as st
import io
import requests

# -----------------------------
# Global Column Rename Map
# -----------------------------
rename_map = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
    'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
    'P2O5 [wt%]': 'P2O5', 'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr',
    'Y [ppm]': 'Y', 'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb', 'Dy [ppm]': 'Dy',
    'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm', 'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
    'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
}

# -----------------------------
# Function Definitions
# -----------------------------
def load_data(upload):
    df = pd.read_csv(upload, low_memory=False)
    df.rename(columns=rename_map, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    # Keep only 45–57 wt% SiO2
    return df.loc[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    # Major and trace-element ratios
    df['Th/La'] = df['Th'] / (df['La'] + eps)
    df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr'] = df['Nb'] / (df['Zr'] + eps)
    df['La/Zr'] = df['La'] / (df['Zr'] + eps)
    df['La/Yb'] = df['La'] / (df['Yb'] + eps)
    df['Sr/Y']  = df['Sr'] / (df['Y'] + eps)
    df['Gd/Yb'] = df['Gd'] / (df['Yb'] + eps)
    df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb'] = df['Dy'] / (df['Yb'] + eps)
    df['Th/Yb'] = df['Th'] / (df['Yb'] + eps)
    df['Nb/Yb'] = df['Nb'] / (df['Yb'] + eps)
    df['Zr/Y']  = df['Zr'] / (df['Y'] + eps)
    df['Ti/Zr'] = df['Ti'] / (df['Zr'] + eps)
    df['Y/Nb']  = df['Y'] / (df['Nb'] + eps)
    df['Th/Nb'] = df['Th'] / (df['Nb'] + eps)
    # Optional Ti ratios
    if 'V' in df.columns:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Al' in df.columns:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)
    return df

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt and Basaltic Andesite Geochemistry Interpreter")
st.write("Upload your own GEOROC CSV parts, or auto-load the JETOA basalt suite from GitHub:")

uploaded_files = st.file_uploader(
    "Upload split GEOROC parts (CSV only)",
    type=["csv"],
    accept_multiple_files=True
)
use_example = st.checkbox("Use cleaned GEOROC dataset example")

def fetch_jetoa_parts():
    api_url = "https://api.github.com/repos/holderds/BasaltChem/contents"
    r = requests.get(api_url)
    r.raise_for_status()
    entries = r.json()
    names = [
        e["name"]
        for e in entries
        if e["name"].startswith("2024-12-2JETOA_") and e["name"].endswith(".csv")
    ]
    return [
        f"https://raw.githubusercontent.com/holderds/BasaltChem/main/{n}"
        for n in sorted(names)
    ]

# -----------------------------
# Load Data
# -----------------------------
if use_example:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt.csv",
        low_memory=False
    )
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)

elif uploaded_files:
    df_parts, skipped = [], []
    for f in uploaded_files:
        try:
            part = load_data(f)
            part = part.apply(pd.to_numeric, errors="ignore")
            if "SiO2" not in part.columns:
                skipped.append(f.name)
                continue
            df_parts.append(part)
        except Exception:
            skipped.append(f.name)
    if not df_parts:
        st.error("❌ No valid uploaded CSVs found.")
        st.stop()
    df = pd.concat(df_parts, ignore_index=True)
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)
    if skipped:
        st.warning("⚠ Skipped: " + ", ".join(skipped))

else:
    st.info("🔄 Loading JETOA basalt training data from GitHub…")
    urls = fetch_jetoa_parts()
    df_parts, skipped = [], []
    for url in urls:
        try:
            txt = requests.get(url).text
            part = pd.read_csv(io.StringIO(txt), low_memory=False)
            part.rename(columns=rename_map, inplace=True)
            part = part.apply(pd.to_numeric, errors="ignore")
            if "SiO2" not in part.columns:
                skipped.append(url)
                continue
            df_parts.append(part)
        except Exception:
            skipped.append(url)
    if not df_parts:
        st.error("❌ Could not load any remote parts.")
        st.stop()
    df = pd.concat(df_parts, ignore_index=True)
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)
    if skipped:
        st.warning("⚠ Skipped remote files:
" + "
".join(skipped))

# -----------------------------
# Configuration for Plots
# -----------------------------
CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 0.613, 'Pr': 0.090, 'Nd': 0.459,
    'Sm': 0.153, 'Eu': 0.057, 'Gd': 0.199, 'Tb': 0.036,
    'Dy': 0.254, 'Ho': 0.050, 'Er': 0.165, 'Tm': 0.028,
    'Yb': 0.165, 'Lu': 0.025
}

REE_ELEMENTS = [
    'La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'
]

# -----------------------------
# εNd vs εHf Plot (if available)
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd")
    ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf Isotope Plot")
    st.pyplot(fig)
    st.session_state['last_figure'] = fig

# -----------------------------
# REE Spider Plot
if all(elem in df.columns for elem in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e] / CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, normed, alpha=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel("Normalized to Chondrite")
    ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)
    st.session_state['last_figure'] = fig2

# -----------------------------
# Preview Computed Ratios
ratio_cols = [
    'Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb',
    'Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb'
]
existing = [c for c in ratio_cols if c in df.columns]
if existing:
    st.subheader("Preview of Computed Ratios")
    st.dataframe(df[existing].describe().T.style.format(precision=2))

# -----------------------------
# Download Buttons
if st.button("📥 Download Results CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="geochem_results.csv")

if 'last_figure' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_figure'].savefig(buf, format="png")
    st.download_button("📷 Download Last Plot", buf.getvalue(), "plot.png", "image/png")

# -----------------------------
# Supervised Classification
st.subheader("Supervised Classification (Tectonic Setting)")
all_features = df.select_dtypes(include=[np.number]).columns.tolist()
label_options = [c for c in df.columns if df[c].nunique() < 20]
label_col = st.selectbox("Label column", label_options)
features = st.multiselect("Features", all_features, default=all_features[:10])

if features and label_col:
    cls = df.dropna(subset=features+[label_col])
    X = cls[features]
    y = cls[label_col].astype('category').cat.codes
    if len(X) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = MLPClassifier((50,25), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        st.text("Classification Report:")
        st.text(report)
    else:
        st.warning("Not enough samples for training/test split.")
