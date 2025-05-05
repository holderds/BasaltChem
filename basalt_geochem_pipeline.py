# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Configuration
# -----------------------------
GITHUB_API_URL = "https://api.github.com/repos/holderds/BasaltChem/contents"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
CSV_PREFIX = "2024-12-2JETOA"

# Rename map for GEOROC‐style CSVs
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
    'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
    'P2O5 [wt%]': 'P2O5', 'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr',
    'Y [ppm]': 'Y', 'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb', 'Dy [ppm]': 'Dy',
    'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm', 'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
    'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr',
    'Hf [ppm]': 'Hf', 'Ta [ppm]': 'Ta',
    '143Nd/144Nd': '143Nd/144Nd', '176Hf/177Hf': '176Hf/177Hf',
    '87Sr/86Sr': '87Sr/86Sr', '206Pb/204Pb': '206Pb/204Pb', '207Pb/204Pb': '207Pb/204Pb'
}

# Chondrite normalizing values for REE spider plot
CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 1.59, 'Pr': 0.51, 'Nd': 1.23, 'Sm': 0.28,
    'Eu': 0.12, 'Gd': 0.27, 'Tb': 0.036, 'Dy': 0.26, 'Ho': 0.053,
    'Er': 0.17, 'Tm': 0.026, 'Yb': 0.17, 'Lu': 0.025
}

# -----------------------------
# Utility Functions
# -----------------------------
def load_data(source):
    """Read a CSV (local or raw text), rename columns."""
    df = pd.read_csv(source, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only samples with 45–57 wt% SiO2."""
    if 'SiO2' not in df.columns:
        return df.iloc[0:0]
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute all element ratios, anomalies, and ε-values."""
    eps = 1e-6
    # Anomalies
    if all(c in df for c in ('La','Ce','Pr')):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    if all(c in df for c in ('Sm','Eu','Gd')):
        df['Eu/Eu*'] = df['Eu'] / (np.sqrt(df['Sm'] * df['Gd']) + eps)
    if all(c in df for c in ('Zr','Ta','Nb')):
        df['Nb/Nb*'] = df['Nb'] / (np.sqrt(df['Zr'] * df['Ta']) + eps)
    # Simple ratios
    for num, den, name in [
        ('Th','La','Th/La'), ('Nb','Zr','Nb/Zr'), ('La','Zr','La/Zr'),
        ('La','Yb','La/Yb'), ('Sr','Y','Sr/Y'), ('Gd','Yb','Gd/Yb'),
        ('Dy','Yb','Dy/Yb'), ('Th','Nb','Th/Nb'), ('Nb','Yb','Nb/Yb'),
        ('Zr','Y','Zr/Y'), ('Ti','Zr','Ti/Zr'), ('Y','Nb','Y/Nb'),
        ('Ti','V','Ti/V'), ('Ti','Al','Ti/Al'), ('Sm','Nd','Sm/Nd')
    ]:
        if num in df and den in df:
            df[name] = df[num] / (df[den] + eps)
    # ε-values
    if '143Nd/144Nd' in df:
        CHUR_Nd = 0.512638
        df['εNd'] = (df['143Nd/144Nd'] - CHUR_Nd) / CHUR_Nd * 1e4
    if '176Hf/177Hf' in df:
        CHUR_Hf = 0.282785
        df['εHf'] = (df['176Hf/177Hf'] - CHUR_Hf) / CHUR_Hf * 1e4
    if all(c in df for c in ('206Pb/204Pb','207Pb/204Pb')):
        df['Pb_iso'] = df['206Pb/204Pb'] / df['207Pb/204Pb']
    return df

def preprocess_list(df_list):
    df = pd.concat(df_list, ignore_index=True)
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)
    return df

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(layout="wide")
st.title("Basalt Geochemistry Interpreter")

st.sidebar.header("Data Input")
use_github = st.sidebar.checkbox("Load training data from GitHub")
uploaded_files = st.sidebar.file_uploader("Upload CSV parts", type="csv", accept_multiple_files=True)

df = None
skipped = []

# --- GitHub loading block ---
if use_github:
    r = requests.get(GITHUB_API_URL)
    if r.status_code != 200:
        st.error("❌ Failed to fetch GitHub contents")
        st.stop()
    repo_files = r.json()
    # filter for CSVs with the right prefix
    csv_names = [f["name"] for f in repo_files if f["name"].startswith(CSV_PREFIX) and f["name"].endswith(".csv")]
    if not csv_names:
        st.error("❌ No training CSVs found in repo")
        st.stop()
    parts = []
    for name in csv_names:
        url = GITHUB_RAW_URL + name
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            part = pd.read_csv(StringIO(resp.text), low_memory=False)
            part.rename(columns=RENAME_MAP, inplace=True)
            parts.append(part)
        except Exception:
            skipped.append(name)
    if not parts:
        st.error("❌ Could not load any remote parts")
        st.stop()
    df = preprocess_list(parts)
    if skipped:
        st.warning(f"Skipped {len(skipped)} remote files: {', '.join(skipped)}")

# --- Upload block ---
elif uploaded_files:
    parts = []
    for f in uploaded_files:
        try:
            parts.append(load_data(f))
        except Exception:
            skipped.append(f.name)
    if not parts:
        st.error("❌ Could not load any uploaded files")
        st.stop()
    df = preprocess_list(parts)
    if skipped:
        st.warning(f"Skipped {len(skipped)} uploads: {', '.join(skipped)}")

else:
    st.info("Upload CSV parts or select GitHub to begin")
    st.stop()

# -----------------------------
# Plots & Summaries
# -----------------------------
if 'εNd' in df and 'εHf' in df:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf"); ax.set_title("εNd vs εHf")
    st.pyplot(fig)

# REE Spider Plot
ree_cols = [e for e in CHONDRITE_VALUES if e in df]
if ree_cols:
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e]/CHONDRITE_VALUES[e] for e in ree_cols]
        ax2.plot(ree_cols, normed, alpha=0.5)
    ax2.set_yscale('log'); ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)

st.subheader("Computed Ratios Summary")
ratio_cols = [c for c in df.columns if '/' in c and pd.api.types.is_numeric_dtype(df[c])]
if ratio_cols:
    st.dataframe(df[ratio_cols].describe().T)
else:
    st.warning("No ratios computed—check that your data contain required elements.")

# -----------------------------
# ML Classification
# -----------------------------
st.subheader("ML Classification: Tectonic Setting")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
label_col = st.selectbox("Select label column", [c for c in df if df[c].nunique()<20])
features = st.multiselect("Select features", numeric_cols, default=numeric_cols)

if features and label_col:
    cls = df.dropna(subset=features+[label_col])
    X = cls[features]
    y = cls[label_col].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = (y_pred == y_test).mean()
    st.text(f"Accuracy: {acc:.2f}")
    st.dataframe(pd.DataFrame(report).T)

# -----------------------------
# Download processed data
# -----------------------------
st.sidebar.subheader("Export")
if st.sidebar.button("Download processed CSV"):
    csv = df.to_csv(index=False).encode()
    st.sidebar.download_button("📥 Download CSV", data=csv, file_name="processed_geochem_data.csv")
