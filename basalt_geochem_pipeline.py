# basalt_geochem_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

# ─────────────────────────────────────────────────────────────────────────────
# 1) DATA LOADING / CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_like):
    """Read, rename columns, coerce to numeric."""
    df = pd.read_csv(csv_like, low_memory=False)
    rename_map = {
        'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5', 'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr',
        'Y [ppm]': 'Y', 'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb', 'Dy [ppm]': 'Dy',
        'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm', 'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
        'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr',
        'Hf [ppm]': 'Hf', 'Sr [ppm]': 'Sr'
    }
    df.rename(columns=rename_map, inplace=True)
    # coerce everything numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def list_github_csvs():
    """List all CSV files in the root of your repo prefixed with '2024-12-2JETOA'."""
    api = "https://api.github.com/repos/holderds/BasaltChem/contents"
    resp = requests.get(api)
    resp.raise_for_status()
    files = resp.json()
    return sorted(
        f["name"] for f in files
        if f["type"] == "file" and
           f["name"].startswith("2024-12-2JETOA") and
           f["name"].endswith(".csv")
    )

@st.cache_data(show_spinner=False)
def load_github_data():
    parts = list_github_csvs()
    dfs, skipped = [], []
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    for name in parts:
        try:
            txt = requests.get(base + name).text
            dfs.append(load_data(StringIO(txt)))
        except Exception:
            skipped.append(name)
    return dfs, skipped

# UI for data source
st.write("### Data source")
use_github = st.checkbox("🔄 Load all JETOA CSVs from GitHub")
uploaded = st.file_uploader("…or upload your own CSVs", accept_multiple_files=True, type="csv")

if use_github:
    df_parts, skipped = load_github_data()
    if skipped:
        st.warning(f"Skipped {len(skipped)} files: {', '.join(skipped)}")
elif uploaded:
    df_parts, skipped = [], []
    for f in uploaded:
        try:
            df_parts.append(load_data(f))
        except:
            skipped.append(f.name)
    if skipped:
        st.warning(f"Skipped {len(skipped)} uploads: {', '.join(skipped)}")
else:
    st.info("Please upload CSVs or select GitHub option.")
    st.stop()

if not df_parts:
    st.error("❌ No valid data loaded.")
    st.stop()

df = pd.concat(df_parts, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2) FILTER & RATIOS
# ─────────────────────────────────────────────────────────────────────────────

def filter_basaltic(df):
    if "SiO2" not in df.columns:
        st.error("SiO2 column missing—check your data.")
        st.stop()
    return df.loc[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)].reset_index(drop=True)

def compute_ratios(df):
    eps = 1e-6
    # anomaly formulas
    df["Th/La"]   = df["Th"]   / (df["La"]   + eps)
    df["Ce/Ce*"]  = df["Ce"]   / (np.sqrt(df["La"]*df["Pr"]) + eps)
    df["Eu/Eu*"]  = df["Eu"]   / (np.sqrt(df["Sm"]*df["Gd"]) + eps)
    df["Nb/Nb*"]  = df["Nb"]   / (np.sqrt(df["Zr"]*df["Y"])  + eps)
    # common ratios
    df["Nb/Zr"]   = df["Nb"]   / (df["Zr"]   + eps)
    df["La/Yb"]   = df["La"]   / (df["Yb"]   + eps)
    df["Dy/Yb"]   = df["Dy"]   / (df["Yb"]   + eps)
    df["Sr/Y"]    = df["Sr"]   / (df["Y"]    + eps)
    df["Gd/Yb"]   = df["Gd"]   / (df["Yb"]   + eps)
    df["Nd/Nd*"]  = df["Nd"]   / (np.sqrt(df["Pr"]*df["Sm"]) + eps)
    # isotopes if present
    if "143Nd/144Nd" in df: 
        CHUR_Nd = 0.512638
        df["εNd"] = ((df["143Nd/144Nd"] - CHUR_Nd) / CHUR_Nd)*1e4
    if "176Hf/177Hf" in df: 
        CHUR_Hf = 0.282785
        df["εHf"] = ((df["176Hf/177Hf"] - CHUR_Hf) / CHUR_Hf)*1e4
    if {"206Pb/204Pb","207Pb/204Pb"} <= set(df.columns):
        df["Pb_iso"] = df["206Pb/204Pb"] / df["207Pb/204Pb"]
    return df

df = filter_basaltic(df)
df = compute_ratios(df)

# ─────────────────────────────────────────────────────────────────────────────
# 3) PLOTS
# ─────────────────────────────────────────────────────────────────────────────

st.write("## Geochemical Plots")

# εNd vs εHf
if {"εNd","εHf"} <= set(df.columns):
    fig,ax=plt.subplots()
    ax.scatter(df["εNd"], df["εHf"], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf"); ax.set_title("εNd vs εHf")
    st.pyplot(fig)

# REE Spider
REE = ["La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"]
CHON = {"La":0.248,"Ce":0.612,"Pr":0.098,"Nd":0.445,"Sm":0.153,"Eu":0.056,
        "Gd":0.199,"Tb":0.027,"Dy":0.156,"Ho":0.035,"Er":0.103,"Tm":0.015,
        "Yb":0.132,"Lu":0.020}
if set(REE) <= set(df.columns):
    fig,ax=plt.subplots()
    for _,r in df.iterrows():
        norm = [r[e]/CHON[e] for e in REE]
        ax.plot(REE,norm,alpha=0.4)
    ax.set_yscale("log"); ax.set_title("REE Spider (Normalized)")
    st.pyplot(fig)

# Ratios summary
ratios = [c for c in ["Th/La","Ce/Ce*","Eu/Eu*","Nb/Nb*","Nb/Zr","La/Yb","Dy/Yb","Sr/Y","Gd/Yb","Nd/Nd*"] if c in df]
if ratios:
    st.write("### Ratio summary")
    st.dataframe(df[ratios].describe().T.style.format("{:.2f}"))
else:
    st.warning("No ratios computed—check that your data contain the required elements.")

# ─────────────────────────────────────────────────────────────────────────────
# 4) ML CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

st.write("## Supervised Classification (Tectonic Setting)")

# label choices (<20 categories)
label = st.selectbox("Label column", [c for c in df.columns if df[c].nunique() < 20])
# all numeric features
nums  = df.select_dtypes(float).columns.tolist()
# default to first 10
feats = st.multiselect("Features", options=nums, default=nums[:10])

if label and feats:
    cls = df.dropna(subset=feats+[label])
    if len(cls) < 2:
        st.error("Not enough data for training.")
    else:
        X = cls[feats]; y = cls[label].astype("category").cat.codes
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
        model = MLPClassifier((50,25),max_iter=500,random_state=42)
        model.fit(Xtr,ytr)
        ypr = model.predict(Xte)
        from sklearn.metrics import classification_report
        st.text("Classification Report")
        st.text(classification_report(yte,ypr,zero_division=0))

        # show probabilities
        proba = model.predict_proba(Xte)
        dfp = pd.DataFrame(proba, columns=model.classes_)
        st.write("Prediction probabilities:")
        st.dataframe(dfp.head())

# ─────────────────────────────────────────────────────────────────────────────
# 5) EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

st.write("## Export")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download processed CSV",data=csv,file_name="processed_basalt.csv",mime="text/csv")
