# basalt_geochem_pipeline.py

import requests, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# --- CONFIGURATION ---
GITHUB_REPO = "holderds/basaltchem"
CSV_PREFIX = "2024-12-2JETOA_"
BRANCH = "main"

# Rename map for GEOROC exports
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
    # isotope columns will drop anyway if missing
}

CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 0.624, 'Pr': 0.088, 'Nd': 0.492,
    'Sm': 0.153, 'Eu': 0.060, 'Gd': 0.256, 'Tb': 0.040,
    'Dy': 0.253, 'Ho': 0.058, 'Er': 0.165, 'Tm': 0.026,
    'Yb': 0.161, 'Lu': 0.025
}

REE_ELEMENTS = list(CHONDRITE_VALUES.keys())
MAJOR_ELEMENTS = ['SiO2','Ti','Al','Fe2O3','MgO','CaO','Na2O','K2O','P2O5']
IMMOBILE = ['Ti','Al','Zr','Nb','Y','Th','Sc','Co','Ni','Cr','Hf']
ANOMALY_RATIOS = ['Ce/Ce*','Eu/Eu*','Nb/Nb*']

# --- UTILITIES ---

def list_github_csvs():
    """Query GitHub API for all raw CSV URLs with our prefix."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    csv_paths = [e['path'] for e in tree
                 if e['path'].startswith(CSV_PREFIX) and e['path'].endswith(".csv")]
    return [
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{p}"
        for p in csv_paths
    ]

def load_and_clean(url):
    """Load one CSV, rename, coerce numeric, drop empty cols."""
    df = pd.read_csv(url, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce everything to numeric if possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop columns that are now entirely NaN
    df.dropna(axis=1, how='all', inplace=True)
    return df

def filter_basalt(df):
    if 'SiO2' not in df.columns:
        raise KeyError("SiO2 column missing")
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    # standard ratios
    df['Th/La']   = df['Th']  / (df['La']  + eps)
    df['Ce/Ce*']  = df['Ce']  / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Eu/Eu*']  = df['Eu']  / (np.sqrt(df['Sm'] * df['Gd']) + eps)
    df['Nb/Nb*']  = df['Nb']  / (np.sqrt(df['Zr'] * df['Y']) + eps)
    df['La/Yb']   = df['La']  / (df['Yb']  + eps)
    df['Dy/Yb']   = df['Dy']  / (df['Yb']  + eps)
    df['Sr/Y']    = df['Sr']  / (df['Y']   + eps) if 'Sr' in df.columns else np.nan
    # ε‐values
    if '143Nd/144Nd' in df and '176Hf/177Hf' in df:
        CHUR_Nd = 0.512638; CHUR_Hf = 0.282785
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR_Nd)/CHUR_Nd)*1e4
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR_Hf)/CHUR_Hf)*1e4
    # Pb‐ratio
    if '206Pb/204Pb' in df and '207Pb/204Pb' in df:
        df['Pb_iso'] = df['206Pb/204Pb']/df['207Pb/204Pb']
    return df

# --- STREAMLIT APP ---

st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("🔄 Loading training CSVs from GitHub…")
urls = list_github_csvs()
if not urls:
    st.error("❌ No training CSVs found on GitHub with prefix.")
    st.stop()

dfs = []
for u in urls:
    try:
        part = load_and_clean(u)
        dfs.append(part)
    except Exception as e:
        st.warning(f"Skipped {u.split('/')[-1]}: {e}")

if not dfs:
    st.error("❌ Could not load any valid parts.")
    st.stop()

# concatenate, filter, compute
df = pd.concat(dfs, ignore_index=True)
try:
    df = filter_basalt(df)
except KeyError:
    st.error("SiO2 column missing—check your data.")
    st.stop()
df = compute_ratios(df)

# --- PLOTS ---

if {'εNd','εHf'}.issubset(df.columns):
    fig,ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set(xlabel="εNd",ylabel="εHf",title="εNd vs εHf")
    st.pyplot(fig)

if set(REE_ELEMENTS).issubset(df.columns):
    fig,ax = plt.subplots()
    for _,r in df.iterrows():
        norm = [r[e]/CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax.plot(REE_ELEMENTS, norm, alpha=0.3)
    ax.set(yscale='log',ylabel='Chondrite-normalized',title='REE Spider')
    st.pyplot(fig)

# --- RATIO SUMMARY ---

ratios = [r for r in ['Th/La','Ce/Ce*','Eu/Eu*','Nb/Nb*','La/Yb','Dy/Yb','Sr/Y','Pb_iso'] if r in df]
if ratios:
    st.subheader("Computed Ratios")
    st.dataframe(df[ratios].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check that your data contain the required elements.")

# --- CLASSIFICATION ---

st.subheader("Supervised Classification (Tectonic Setting)")
label_col = st.selectbox("Label column (categorical)", [c for c in df.columns if df[c].nunique()<20])
feature_cols = MAJOR_ELEMENTS + REE_ELEMENTS + ratios
# only keep those actually in df
feature_cols = [c for c in feature_cols if c in df.columns]
selected = st.multiselect("Features", feature_cols, default=feature_cols[:10])

if selected and label_col:
    clean = df.dropna(subset=selected+[label_col])
    X, y = clean[selected], clean[label_col].astype('category').cat.codes
    if len(X)>1:
        Xtr,Xt, ytr,yt = train_test_split(X,y,test_size=0.3,random_state=42)
        clf = MLPClassifier((50,25),max_iter=500,random_state=42).fit(Xtr,ytr)
        pred = clf.predict(Xt)
        st.write("Accuracy:", np.mean(pred==yt))
    else:
        st.warning("Not enough data for classification.")

# --- DOWNLOAD ---

if st.button("Download cleaned+ratios CSV"):
    out = df.to_csv(index=False).encode()
    st.download_button("📥 Download data", data=out, file_name="basaltchem_training_cleaned.csv", mime="text/csv")
