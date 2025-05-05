# basalt_geochem_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

# -----------------------------
# Utility functions
# -----------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows/cols, sanitize column names, coerce to numeric."""
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    cols = (
        df.columns
          .str.replace(r"\[.*\]", "", regex=True)
          .str.replace(r"[^0-9A-Za-z_/\.]", "_", regex=True)
          .str.replace("/", "_")
          .str.strip()
    )
    df.columns = cols
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def filter_basalt(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 45–57 wt% SiO2."""
    return df.loc[(df.SiO2 >= 45) & (df.SiO2 <= 57)]

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom geochemical ratios & anomalies."""
    eps = 1e-6
    # Basic ratios
    df["Th_La"]   = df.Th  / (df.La  + eps)
    df["Ce_Ce_star"] = df.Ce  / (np.sqrt(df.La * df.Pr) + eps)
    df["Eu_Eu_star"] = df.Eu  / (np.sqrt(df.Sm * df.Gd) + eps)
    df["Nb_Nb_star"] = df.Nb  / (np.sqrt(df.Zr * df.Yb) + eps)
    # Trace element ratios
    for a,b in [("Nb","Zr"),("La","Zr"),("La","Yb"),("Sr","Y"),("Gd","Yb"),
                ("Dy","Yb"),("Th","Yb"),("Nb","Yb"),("Zr","Y"),("Ti","Zr"),
                ("Y","Nb"),("Th","Nb"),("Ti","V"),("Ti","Al")]:
        if a in df and b in df:
            df[f"{a}_{b}"] = df[a] / (df[b] + eps)
    # Isotope ε-values
    if "143Nd_144Nd" in df:
        CHUR_Nd = 0.512638
        df["eNd"] = ((df["143Nd_144Nd"] - CHUR_Nd)/CHUR_Nd)*1e4
    if "176Hf_177Hf" in df:
        CHUR_Hf = 0.282785
        df["eHf"] = ((df["176Hf_177Hf"] - CHUR_Hf)/CHUR_Hf)*1e4
    if "206Pb_204Pb" in df and "207Pb_204Pb" in df:
        df["Pb_iso_ratio"] = df["206Pb_204Pb"] / df["207Pb_204Pb"]
    return df

def preprocess(sources):
    """
    sources: list of (source, name) where source is either
      - a file-like (uploaded) or
      - a URL string
    Returns (df, skipped_names)
    """
    parts = []
    skipped = []
    for src,name in sources:
        try:
            df = pd.read_csv(src, low_memory=False) if isinstance(src,str) else pd.read_csv(src)
            df = clean_df(df)
            if "SiO2" not in df.columns:
                skipped.append(name); continue
            df = filter_basalt(df)
            df = compute_ratios(df)
            if df.empty:
                skipped.append(name); continue
            parts.append(df)
        except Exception:
            skipped.append(name)
    if not parts:
        st.error("❌ No valid GEOROC parts loaded.")
        st.stop()
    return pd.concat(parts, ignore_index=True), skipped

# -----------------------------
# Data loading UI
# -----------------------------
use_remote = st.checkbox("Load all 2024-12-2JETOA CSV parts from GitHub")
uploads     = st.file_uploader("…or upload your split GEOROC CSV parts here", type="csv", accept_multiple_files=True)

if use_remote:
    # fetch list of all files in repo root via GitHub API
    tree_url = "https://api.github.com/repos/holderds/BasaltChem/git/trees/main?recursive=1"
    try:
        tree = requests.get(tree_url).json()["tree"]
        paths = [item["path"] for item in tree
                 if item["path"].startswith("2024-12-2JETOA_") and item["path"].endswith(".csv")]
        base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
        sources = [(base+p, p) for p in paths]
    except Exception:
        st.error("Failed to fetch file list from GitHub.")
        st.stop()
    df, skipped = preprocess(sources)
    if skipped:
        st.warning(f"⚠️ Skipped: {', '.join(skipped)}")
elif uploads:
    sources = [(f, f.name) for f in uploads]
    df, skipped = preprocess(sources)
    if skipped:
        st.warning(f"⚠️ Skipped uploads: {', '.join(skipped)}")
else:
    st.info("Please upload CSV(s) or select the GitHub option.")
    st.stop()

# -----------------------------
# εNd vs εHf Isotope Plot
# -----------------------------
if {"eNd","eHf"}.issubset(df.columns):
    fig, ax = plt.subplots()
    ax.scatter(df.eNd, df.eHf, alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf"); ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state.last_fig = fig

# -----------------------------
# REE Spider Plot
# -----------------------------
REEs = ["La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
# chondrite normalization factors
CHONDRITE = dict(La=0.237, Ce=0.613, Pr=0.096, Nd=0.448,
                 Sm=0.153, Eu=0.061, Gd=0.368, Tb=0.062,
                 Dy=0.577, Ho=0.096, Er=0.260, Tm=0.038,
                 Yb=0.314, Lu=0.044)

if all(e in df.columns for e in REEs):
    fig2, ax2 = plt.subplots()
    for _,r in df.iterrows():
        normed = [r[e]/CHONDRITE[e] for e in REEs]
        ax2.plot(REEs, normed, alpha=0.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("Normalized to Chondrite"); ax2.set_title("REE Spider Diagram")
    st.pyplot(fig2)
    st.session_state.last_fig = fig2

# -----------------------------
# Ratios summary
# -----------------------------
ratio_cols = [c for c in df.columns if "_" in c and c not in ["eNd","eHf"]]
st.subheader("Computed Ratios Summary")
if ratio_cols:
    desc = df[ratio_cols].describe().T.round(2)
    st.dataframe(desc)
else:
    st.warning("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# Export & Download
# -----------------------------
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Combined Data CSV", csv_bytes, "basalt_geochem_results.csv")

if "last_fig" in st.session_state:
    buf = io.BytesIO()
    st.session_state.last_fig.savefig(buf, format="png")
    st.download_button("📷 Download Last Plot as PNG", buf.getvalue(), "plot.png", "image/png")

# -----------------------------
# ML Classification
# -----------------------------
st.subheader("ML Classification (Tectonic Setting)")
label_col = st.selectbox("Select Label Column", [c for c in df.columns if df[c].nunique()<20])
features  = st.multiselect("Select Features", df.select_dtypes(float).columns.tolist())

if st.button("Run Classification") and label_col and features:
    # prepare
    dfc = df.dropna(subset=features+[label_col])
    X = dfc[features]; y = dfc[label_col].astype("category").cat.codes
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
    model = MLPClassifier((50,25), max_iter=500, random_state=42)
    model.fit(Xtr,ytr)
    yp = model.predict(Xte)
    report = classification_report(yte, yp, output_dict=True, zero_division=0)
    st.text("Classification report")
    st.dataframe(pd.DataFrame(report).T.round(2))
