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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

GITHUB_USER = "holderds"
GITHUB_REPO = "basaltchem"
PREFIX      = "2024-12-2JETOA"
ESSENTIAL_SI = "SiO2"

# raw→clean column mapping
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al',
    'Fe2O3(t) [wt%]': 'Fe2O3', 'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
    'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O', 'P2O5 [wt%]': 'P2O5',
    **{f"{el} [ppm]": el for el in [
        'Sr','Th','Nb','Zr','Y','La','Ce','Pr','Nd','Sm','Eu',
        'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','V','Sc','Co','Ni','Cr','Hf'
    ]},
    '87Sr/86Sr': 'Sr87_86',
    '206Pb/204Pb':'Pb206_204','207Pb/204Pb':'Pb207_204',
    '143Nd/144Nd':'Nd143_144','176Hf/177Hf':'Hf176_177'
}

REE_LIST = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
CHONDRITE = {
    'La':0.237,'Ce':1.59,'Pr':0.49,'Nd':1.23,'Sm':0.28,
    'Eu':0.11,'Gd':0.27,'Tb':0.04,'Dy':0.26,'Ho':0.05,
    'Er':0.17,'Tm':0.02,'Yb':0.17,'Lu':0.03
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def list_github_csvs(prefix):
    """List all .csv files in the repo whose path starts with prefix"""
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/main?recursive=1"
    resp = requests.get(url)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])
    return [item["path"] for item in tree
            if item["path"].startswith(prefix) and item["path"].endswith(".csv")]

def load_and_clean(source):
    """
    Read everything as string → rename → coerce to numeric → drop missing SiO2.
    Accepts local file handles or URLs.
    """
    df = pd.read_csv(source, dtype=str)
    df.rename(columns=RENAME_MAP, inplace=True)
    for clean in RENAME_MAP.values():
        if clean in df.columns:
            df[clean] = pd.to_numeric(df[clean], errors="coerce")
    df.dropna(subset=[ESSENTIAL_SI], inplace=True)
    return df

def filter_si(df):
    """Keep only basalt & basaltic andesite (45–57 wt% SiO₂)."""
    return df[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    def R(a,b,name):
        if a in df.columns and b in df.columns:
            df[name] = df[a] / (df[b] + eps)
    # major/trace
    R("Th","La","Th/La")
    if all(c in df.columns for c in ["La","Pr","Ce"]):
        df["Ce/Ce*"] = df["Ce"] / np.sqrt(df["La"]*df["Pr"] + eps)
    R("Nb","Zr","Nb/Zr"); R("La","Zr","La/Zr"); R("La","Yb","La/Yb")
    R("Sr","Y","Sr/Y"); R("Gd","Yb","Gd/Yb"); 
    if all(c in df.columns for c in ["Nd","Pr","Sm"]):
        df["Nd/Nd*"] = df["Nd"]/np.sqrt(df["Pr"]*df["Sm"]+eps)
    R("Dy","Yb","Dy/Yb"); R("Th","Yb","Th/Yb"); R("Nb","Yb","Nb/Yb")
    R("Zr","Y","Zr/Y"); R("Ti","V","Ti/V"); R("Ti","Al","Ti/Al")
    R("Sm","Nd","Sm/Nd")
    # isotopic
    if "Nd143_144" in df: df["εNd"] = (df["Nd143_144"]-0.512638)/0.512638*1e4
    if "Hf176_177" in df: df["εHf"] = (df["Hf176_177"]-0.282785)/0.282785*1e4
    # composite
    df["Crustal_Thickness_Index"]    = df.get("Dy/Yb")
    df["Melting_Depth_Index"]        = df.get("La/Yb")
    df["Crustal_Contamination_Index"]= df.get("Th/Nb")
    df["Mantle_Fertility_Index"]     = df.get("Nb/Zr")
    return df

def preprocess(parts):
    full = pd.concat(parts, ignore_index=True)
    return compute_ratios(filter_si(full))

# ──────────────────────────────────────────────────────────────────────────────
# 1) INPUT DATA
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("1) Input GEOROC Data")
use_remote = st.checkbox(f"Load all `{PREFIX}*.csv` from GitHub")
uploads    = st.file_uploader("…or upload your CSV parts", type="csv",
                              accept_multiple_files=True)

if use_remote:
    paths = list_github_csvs(PREFIX)
    if not paths:
        st.error("❌ No matching CSVs found on GitHub.")
        st.stop()
    parts, skipped = [], []
    for p in paths:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{p}"
        try:
            parts.append(load_and_clean(url))
        except Exception:
            skipped.append(p)
    if not parts:
        st.error("❌ Could not load any remote parts.")
        st.stop()
    st.write(f"✅ Loaded {len(parts)}; skipped {len(skipped)}: {skipped}")
    df = preprocess(parts)

elif uploads:
    parts, skipped = [], []
    for f in uploads:
        try:
            parts.append(load_and_clean(f))
        except Exception:
            skipped.append(f.name)
    if not parts:
        st.error("❌ No valid uploads.")
        st.stop()
    st.write(f"✅ Loaded {len(parts)} uploads; skipped {len(skipped)}: {skipped}")
    df = preprocess(parts)

else:
    st.info("Please either check **Load from GitHub** or upload CSVs above.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 2) εNd vs εHf Plot
# ──────────────────────────────────────────────────────────────────────────────
if {"εNd","εHf"}.issubset(df.columns):
    fig,ax=plt.subplots()
    ax.scatter(df["εNd"], df["εHf"], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf"); ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state["last_fig"]=fig

# ──────────────────────────────────────────────────────────────────────────────
# 3) REE Spider Plot
# ──────────────────────────────────────────────────────────────────────────────
if all(e in df.columns for e in REE_LIST):
    fig2,ax2=plt.subplots()
    for _,r in df.iterrows():
        norm=[r[e]/CHONDRITE[e] for e in REE_LIST]
        ax2.plot(REE_LIST,norm,alpha=0.3)
    ax2.set_yscale("log"); ax2.set_title("REE Spider Plot")
    st.pyplot(fig2); st.session_state["last_fig"]=fig2

# ──────────────────────────────────────────────────────────────────────────────
# 4) Ratios Summary
# ──────────────────────────────────────────────────────────────────────────────
ratio_cols=[c for c in df.columns if "/" in c or c.startswith("ε")]
if ratio_cols:
    st.subheader("Computed Ratios & ε-Values")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))
else:
    st.error("No ratios computed—check your data contain the required elements.")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Export
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Download results CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download", data=csv, file_name="basalt_results.csv")

if "last_fig" in st.session_state:
    buf=io.BytesIO()
    st.session_state["last_fig"].savefig(buf,format="png")
    st.download_button("📷 Last plot", data=buf.getvalue(),
                       file_name="last_plot.png", mime="image/png")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Supervised Classification
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Supervised Classification (Tectonic Setting)")
label = st.selectbox("Label column", [c for c in df.columns if df[c].nunique()<20])
feats = st.multiselect("Features", df.select_dtypes("number").columns.tolist())

if label and feats:
    dfc = df.dropna(subset=[label]+feats)
    X = dfc[feats]; y = dfc[label].astype("category").cat.codes
    Xtr,Xt, ytr,yt = train_test_split(X,y,test_size=0.3,random_state=42)
    mdl = MLPClassifier((50,25),max_iter=500,random_state=42)
    mdl.fit(Xtr,ytr); yp = mdl.predict(Xt)
    st.text("Classification Report"); st.text(classification_report(yt,yp,zero_division=0))
    df["Predicted"] = None
    df.loc[dfc.index,"Predicted"] = mdl.predict(dfc[feats])
