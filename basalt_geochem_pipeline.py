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

st.set_page_config(layout="wide")
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

# -----------------------------
# CONFIGURATION
# -----------------------------
GITHUB_USER   = "holderds"
GITHUB_REPO   = "basaltchem"
GEOROC_PREFIX = "2024-12-2JETOA"
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
    # trace & REE
    **{f'{el} [ppm]': el for el in [
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
ESSENTIAL_COL = 'SiO2'
REE_ELEMENTS = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
CHONDRITE = {
    'La':0.237,'Ce':1.59,'Pr':0.49,'Nd':1.23,'Sm':0.28,'Eu':0.11,
    'Gd':0.27,'Tb':0.04,'Dy':0.26,'Ho':0.05,'Er':0.17,'Tm':0.02,'Yb':0.17,'Lu':0.03
}

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def list_github_parts(prefix):
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return [
        item["name"]
        for item in data
        if item["name"].startswith(prefix) and item["name"].endswith(".csv")
    ]

def load_and_clean(src):
    """Load CSV (path or URL), rename, coerce to numeric, drop rows without SiO2."""
    df = pd.read_csv(src, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce all renamed columns to numeric
    for col in RENAME_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # drop rows missing essential SiO2
    df = df.dropna(subset=[ESSENTIAL_COL])
    return df

def filter_si(df):
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    # only compute if both cols present
    def safe_ratio(a,b,name):
        if a in df.columns and b in df.columns:
            df[name] = df[a] / (df[b] + eps)

    # major‐trace
    safe_ratio('Th','La','Th/La')
    safe_ratio('Ce',None,'Ce/Ce*')  # special
    if all(c in df.columns for c in ['La','Pr','Ce']):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    safe_ratio('Nb','Zr','Nb/Zr')
    safe_ratio('La','Zr','La/Zr')
    safe_ratio('La','Yb','La/Yb')
    safe_ratio('Sr','Y','Sr/Y')
    safe_ratio('Gd','Yb','Gd/Yb')
    safe_ratio('Nd',None,'Nd/Nd*')
    if all(c in df.columns for c in ['Nd','Pr','Sm']):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    safe_ratio('Dy','Yb','Dy/Yb')
    safe_ratio('Th','Yb','Th/Yb')
    safe_ratio('Nb','Yb','Nb/Yb')
    safe_ratio('Zr','Y','Zr/Y')
    # Ti/V, Ti/Al
    safe_ratio('Ti','V','Ti/V'); safe_ratio('Ti','Al','Ti/Al')
    # Sm/Nd
    safe_ratio('Sm','Nd','Sm/Nd')

    # εNd, εHf
    if 'Nd143_144' in df.columns:
        CHUR = 0.512638
        df['εNd'] = ((df['Nd143_144'] - CHUR) / CHUR) * 1e4
    if 'Hf176_177' in df.columns:
        CHUR = 0.282785
        df['εHf'] = ((df['Hf176_177'] - CHUR) / CHUR) * 1e4

    # composite indices
    safe_ratio('Dy/Yb','', 'Crustal_Thickness_Index')
    df['Melting_Depth_Index'] = df.get('La/Yb', np.nan)
    df['Crustal_Contamination_Index'] = df.get('Th/Nb', np.nan)
    df['Mantle_Fertility_Index']      = df.get('Nb/Zr', np.nan)

    return df

def preprocess(dfs):
    full = pd.concat(dfs, ignore_index=True)
    full = filter_si(full)
    full = compute_ratios(full)
    return full


# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("1) Input GEOROC data")
load_remote = st.checkbox("⬇️ Load all `2024-12-2JETOA` CSVs from GitHub")
uploaded   = st.file_uploader("📂 Or upload one or more CSV parts", type="csv", accept_multiple_files=True)

if load_remote:
    names = list_github_parts(GEOROC_PREFIX)
    dfs   = []
    skipped = []
    for nm in names:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{nm}"
        try:
            part = load_and_clean(url)
            dfs.append(part)
        except Exception:
            skipped.append(nm)
    if not dfs:
        st.error("❌ Could not load any remote parts.")
        st.stop()
    st.write(f"✅ Loaded {len(dfs)} parts; skipped {len(skipped)}: {skipped}")
    df = preprocess(dfs)

elif uploaded:
    dfs = []
    skipped = []
    for f in uploaded:
        try:
            part = load_and_clean(f)
            dfs.append(part)
        except Exception:
            skipped.append(f.name)
    if not dfs:
        st.error("❌ No valid uploads.")
        st.stop()
    st.write(f"✅ Loaded {len(dfs)} uploaded files; skipped {len(skipped)}: {skipped}")
    df = preprocess(dfs)

else:
    st.info("Please either check **Load from GitHub** or upload CSV parts above.")
    st.stop()

# -----------------------------
# 2) εNd vs εHf Plot
# -----------------------------
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state['last_fig'] = fig

# -----------------------------
# 3) REE Spider Plot
# -----------------------------
if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, r in df.iterrows():
        norm = [r[e] / CHONDRITE[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, norm, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# -----------------------------
# 4) Ratios Summary
# -----------------------------
ratio_list = [c for c in df.columns if "/" in c or c.startswith("ε")]
if ratio_list:
    st.subheader("Computed Ratios & ε-Values")
    st.dataframe(df[ratio_list].describe().T.style.format("{:.2f}"))
else:
    st.error("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# 5) Download & Export
# -----------------------------
if st.button("Download full results as CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="basalt_geochem_results.csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Download last plot", data=buf.getvalue(), file_name="last_plot.png", mime="image/png")

# -----------------------------
# 6) Supervised Classification
# -----------------------------
st.subheader("Supervised Classification: Tectonic Setting")
label_col = st.selectbox("Label column", [c for c in df.columns if df[c].nunique() < 20])
features  = st.multiselect("Features (numeric)", df.select_dtypes("number").columns.tolist())

if label_col and features:
    dfc = df.dropna(subset=[label_col]+features)
    X = dfc[features]
    y = dfc[label_col].astype("category").cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MLPClassifier((50,25), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))
    df["Predicted"] = None
    df.loc[dfc.index, "Predicted"] = model.predict(X)

