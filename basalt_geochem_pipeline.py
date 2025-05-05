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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

GITHUB_USER   = "holderds"
GITHUB_REPO   = "basaltchem"
PREFIX        = "2024-12-2JETOA"
ESSENTIAL_COL = "SiO2"

# map raw GEOROC columns → clean names
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2',
    'TiO2 [wt%]': 'Ti',
    'Al2O3 [wt%]': 'Al',
    'Fe2O3(t) [wt%]': 'Fe2O3',
    'MgO [wt%]': 'MgO',
    'CaO [wt%]': 'CaO',
    'Na2O [wt%]': 'Na2O',
    'K2O [wt%]': 'K2O',
    'P2O5 [wt%]': 'P2O5',
    'Sr [ppm]': 'Sr',
    # traces & REEs
    **{f"{el} [ppm]": el for el in [
        'Th','Nb','Zr','Y','La','Ce','Pr','Nd','Sm','Eu',
        'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','V','Sc','Co','Ni','Cr','Hf'
    ]},
    # isotopes
    '87Sr/86Sr': 'Sr87_86',
    '206Pb/204Pb': 'Pb206_204',
    '207Pb/204Pb': 'Pb207_204',
    '143Nd/144Nd': 'Nd143_144',
    '176Hf/177Hf': 'Hf176_177'
}

# for spider-plot normalization
REE_ORDER = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
CHONDRITE = {
    'La':0.237, 'Ce':1.59, 'Pr':0.49, 'Nd':1.23, 'Sm':0.28,
    'Eu':0.11, 'Gd':0.27, 'Tb':0.04, 'Dy':0.26, 'Ho':0.05,
    'Er':0.17, 'Tm':0.02, 'Yb':0.17, 'Lu':0.03
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def list_github_csvs(prefix: str):
    """Return all CSV names in repo root that start with prefix."""
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"
    r   = requests.get(url)
    r.raise_for_status()
    return [
        item['name']
        for item in r.json()
        if item['name'].startswith(prefix) and item['name'].endswith('.csv')
    ]

def load_and_clean(source):
    """
    Load a CSV (path, streamlit file, or URL) *as strings* then 
    rename, coerce to numeric, drop rows missing SiO2.
    """
    # 1) read everything as str to avoid dtype warnings
    df = pd.read_csv(source, dtype=str)
    # 2) rename columns
    df.rename(columns=RENAME_MAP, inplace=True)
    # 3) coerce each renamed column to numeric
    for col in RENAME_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # 4) drop any row missing essential SiO2
    df = df.dropna(subset=[ESSENTIAL_COL])
    return df

def filter_si(df):
    """Keep only basalt (45–57 wt% SiO2)."""
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Add all ratio columns (only if source cols exist)."""
    eps = 1e-6
    def r(a,b,name):
        if a in df.columns and b in df.columns:
            df[name] = df[a] / (df[b] + eps)

    # major‐trace & REE
    r('Th','La','Th/La')
    if all(c in df for c in ['La','Pr','Ce']):
        df['Ce/Ce*'] = df['Ce'] / np.sqrt(df['La']*df['Pr'] + eps)
    r('Nb','Zr','Nb/Zr')
    r('La','Zr','La/Zr')
    r('La','Yb','La/Yb')
    r('Sr','Y','Sr/Y')
    r('Gd','Yb','Gd/Yb')
    if all(c in df for c in ['Nd','Pr','Sm']):
        df['Nd/Nd*'] = df['Nd'] / np.sqrt(df['Pr']*df['Sm'] + eps)
    r('Dy','Yb','Dy/Yb')
    r('Th','Yb','Th/Yb')
    r('Nb','Yb','Nb/Yb')
    r('Zr','Y','Zr/Y')
    r('Ti','V','Ti/V'); r('Ti','Al','Ti/Al')
    r('Sm','Nd','Sm/Nd')
    # isotopic eps
    if 'Nd143_144' in df.columns:
        CH = 0.512638
        df['εNd'] = (df['Nd143_144'] - CH)/CH * 1e4
    if 'Hf176_177' in df.columns:
        CH = 0.282785
        df['εHf'] = (df['Hf176_177'] - CH)/CH * 1e4

    # composite indices (just copy from ratios if present)
    df['Crustal_Thickness_Index']   = df.get('Dy/Yb')
    df['Melting_Depth_Index']       = df.get('La/Yb')
    df['Crustal_Contamination_Index']= df.get('Th/Nb')
    df['Mantle_Fertility_Index']     = df.get('Nb/Zr')
    return df

def preprocess(parts):
    df = pd.concat(parts, ignore_index=True)
    df = filter_si(df)
    df = compute_ratios(df)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 1) INPUT
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("1) Input GEOROC data")
use_remote = st.checkbox("⬇️ Load all `2024-12-2JETOA*` CSVs from GitHub")
uploads    = st.file_uploader("📂 Or upload CSV parts", type="csv", accept_multiple_files=True)

if use_remote:
    names = list_github_csvs(PREFIX)
    parts = []; skipped = []
    for nm in names:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{nm}"
        try:
            parts.append(load_and_clean(url))
        except Exception:
            skipped.append(nm)
    if not parts:
        st.error("❌ Could not load any remote CSVs.")
        st.stop()
    st.write(f"✅ Loaded {len(parts)} remote parts; skipped {len(skipped)}: {skipped}")
    df = preprocess(parts)

elif uploads:
    parts = []; skipped = []
    for f in uploads:
        try:
            parts.append(load_and_clean(f))
        except Exception:
            skipped.append(f.name)
    if not parts:
        st.error("❌ No valid uploaded CSVs.")
        st.stop()
    st.write(f"✅ Loaded {len(parts)} uploads; skipped {len(skipped)}: {skipped}")
    df = preprocess(parts)

else:
    st.info("Please either check **Load from GitHub** or upload CSV files above.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 2) εNd vs εHf
# ──────────────────────────────────────────────────────────────────────────────
if {'εNd','εHf'}.issubset(df.columns):
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf"); ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state['last_fig'] = fig

# ──────────────────────────────────────────────────────────────────────────────
# 3) REE Spider
# ──────────────────────────────────────────────────────────────────────────────
if all(e in df.columns for e in REE_ORDER):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        norm = [row[e]/CHONDRITE[e] for e in REE_ORDER]
        ax2.plot(REE_ORDER, norm, alpha=0.3)
    ax2.set_yscale('log'); ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# ──────────────────────────────────────────────────────────────────────────────
# 4) Ratios Summary
# ──────────────────────────────────────────────────────────────────────────────
cols = [c for c in df.columns if "/" in c or c.startswith("ε")]
if cols:
    st.subheader("Computed Ratios & ε-values")
    st.dataframe(df[cols].describe().T.style.format("{:.2f}"))
else:
    st.error("No ratios computed – check that your data have the required elements.")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Export
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Download full results CSV"):
    out = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download", data=out, file_name="basalt_results.csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Last plot PNG", data=buf.getvalue(),
                       file_name="last_plot.png", mime="image/png")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Supervised Classification
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Supervised Classification: Tectonic Setting")
label = st.selectbox("Label column", [c for c in df.columns if df[c].nunique()<20])
feats = st.multiselect("Features", df.select_dtypes("number").columns.tolist())

if label and feats:
    dfit = df.dropna(subset=[label]+feats)
    X = dfit[feats]
    y = dfit[label].astype("category").cat.codes
    Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.3, random_state=42)
    mdl = MLPClassifier((50,25), max_iter=500, random_state=42)
    mdl.fit(Xtr,ytr)
    yp = mdl.predict(Xtst)
    st.text("Classification Report:")
    st.text(classification_report(ytst,yp,zero_division=0))
    df["Predicted"] = None
    df.loc[dfit.index,"Predicted"] = mdl.predict(dfit[feats])
