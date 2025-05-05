# basalt_geochem_pipeline.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Basalt Geochem Interpreter", layout="wide")

# 1. COLUMN RENAMING MAP
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
    'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
    'P2O5 [wt%]': 'P2O5', 'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr',
    'Y [ppm]': 'Y', 'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb', 'Dy [ppm]': 'Dy',
    'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm', 'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
    'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
}

def load_and_clean(source):
    """
    Read CSV (path, file-like, or text), rename columns, coerce all to numeric,
    drop rows missing SiO2, return cleaned DataFrame.
    """
    if isinstance(source, str) and source.startswith("http"):
        text = requests.get(source).text
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    else:
        df = pd.read_csv(source, low_memory=False)

    # rename
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce all columns except non-numeric identifiers
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop any rows missing silica
    df = df[df['SiO2'].notna()]
    return df

def filter_basaltic(df):
    """Keep only 45–57 wt% SiO2."""
    return df.loc[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute a suite of oxide & trace-element ratios + anomalies."""
    eps = 1e-6
    # classic ratios
    df['Th/La']   = df['Th']  / (df['La']  + eps)
    df['Ce/Ce*']  = df['Ce']  / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']   = df['Nb']  / (df['Zr']  + eps)
    df['La/Zr']   = df['La']  / (df['Zr']  + eps)
    df['La/Yb']   = df['La']  / (df['Yb']  + eps)
    df['Sr/Y']    = df['Sr']  / (df['Y']   + eps)
    df['Gd/Yb']   = df['Gd']  / (df['Yb']  + eps)
    df['Nd/Nd*']  = df['Nd']  / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb']   = df['Dy']  / (df['Yb']  + eps)
    df['Th/Yb']   = df['Th']  / (df['Yb']  + eps)
    df['Nb/Yb']   = df['Nb']  / (df['Yb']  + eps)
    df['Zr/Y']    = df['Zr']  / (df['Y']   + eps)
    df['Ti/Zr']   = df['Ti']  / (df['Zr']  + eps)
    df['Y/Nb']    = df['Y']   / (df['Nb']  + eps)
    # anomaly metrics
    df['Nb/Nb*']  = df['Nb']  / (np.sqrt(df['Zr'] * df['Y']) + eps)
    df['Eu/Eu*']  = df['Eu']  / (np.sqrt(df['Sm'] * df['Gd']) + eps)
    # any Ti‐based ratios
    if 'V' in df and 'Ti' in df:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Al' in df and 'Ti' in df:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)
    return df

st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")
st.write("""
Load **all** `2024-12-2JETOA_*.csv` parts from GitHub as your training set,
or upload your own split CSVs.
""")

use_github = st.checkbox("Fetch training data from GitHub JETOA CSVs")
uploaded = st.file_uploader("Or upload your own GEOROC parts", type="csv", accept_multiple_files=True)

df = None
skipped = []

if use_github:
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    # list every CSV that starts with 2024-12-2JETOA_  (you can trim if you only want basalt)
    filenames = [
        f"2024-12-2JETOA_BASALT_part{i}.csv" for i in range(1,10)
    ] + [
        "2024-12-2JETOA_BASALTIC-ANDESITE_part1.csv",
        "2024-12-2JETOA_BASALTIC-ANDESITE_part2.csv",
    ]
    parts = []
    for fn in filenames:
        url = base + fn
        try:
            part = load_and_clean(url)
            if part.empty:
                skipped.append(fn)
            else:
                parts.append(part)
        except Exception:
            skipped.append(fn)
    if not parts:
        st.error("❌ Could not load any valid JETOA parts from GitHub.")
        st.stop()
    df = pd.concat(parts, ignore_index=True)

elif uploaded:
    parts = []
    for f in uploaded:
        try:
            part = load_and_clean(f)
            if part.empty:
                skipped.append(f.name)
            else:
                parts.append(part)
        except Exception:
            skipped.append(f.name)
    if not parts:
        st.error("❌ Could not load any valid uploaded files.")
        st.stop()
    df = pd.concat(parts, ignore_index=True)

else:
    st.warning("Please either fetch from GitHub or upload CSV parts.")
    st.stop()

if skipped:
    st.warning(f"Skipped {len(skipped)} file(s): {', '.join(skipped)}")

# 45–57 wt% SiO2 filter + ratio compute
df = filter_basaltic(df)
df = compute_ratios(df)

st.subheader("εNd vs εHf (if available)")
if {'εNd','εHf'}.issubset(df):
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set(xlabel="εNd", ylabel="εHf", title="εNd vs εHf")
    st.pyplot(fig)

st.subheader("REE Spider Plot")
REE = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
CHOND = {'La':0.237,'Ce':0.613,'Pr':0.098,'Nd':0.460,'Sm':0.153,'Eu':0.062,'Gd':0.199,'Tb':0.036,'Dy':0.262,'Ho':0.045,'Er':0.140,'Tm':0.020,'Yb':0.160,'Lu':0.025}
if all(e in df for e in REE):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e]/CHOND[e] for e in REE]
        ax2.plot(REE, normed, alpha=0.3)
    ax2.set(yscale='log', ylabel='Chondrite-normalized', title='REE Spider')
    st.pyplot(fig2)

# Preview computed ratios
ratio_cols = [c for c in df.columns if "/" in c and c not in ['εNd','εHf']]
if ratio_cols:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))

# ML classification block (only if you have ≥2 rows)
st.subheader("ML Classification (Tectonic Setting)")
numeric = df.select_dtypes(float).columns.tolist()
features = st.multiselect("Features", numeric, default=numeric[:8])
label = st.selectbox("Label column", [c for c in df.columns if df[c].nunique()<10], index=0)

cls_data = df.dropna(subset=features + [label])
if len(cls_data) >= 2:
    X = cls_data[features]
    y = cls_data[label].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
else:
    st.info("Not enough complete cases for ML training (need ≥2).")

# Export
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download full dataset", data=csv_bytes, file_name="geochem_data.csv", mime="text/csv")
