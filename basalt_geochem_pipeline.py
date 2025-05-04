# basalt_geochem_pipeline.py

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# CONSTANTS & ELEMENT LISTS
# -----------------------------

CHONDRITE_VALUES = {
    'La': 0.234, 'Ce': 1.59,  'Pr': 0.456, 'Nd': 1.23,
    'Sm': 0.298, 'Eu': 0.123, 'Gd': 0.27,  'Tb': 0.1,
    'Dy': 0.26,  'Ho': 0.059, 'Er': 0.167, 'Tm': 0.025,
    'Yb': 0.17,  'Lu': 0.036
}

REE_ELEMENTS = list(CHONDRITE_VALUES.keys())
MAJOR_OXIDES = ['SiO2','Ti','Al','Fe2O3','MgO','CaO','Na2O','K2O','P2O5']

# -----------------------------
# CLEANUP FUNCTION
# -----------------------------

def sanitize_columns(df):
    """
    1) Remove anything in [] or () plus %, wt, ppm, hyphens, whitespace.
    2) Lower-case.
    3) Map to the proper names.
    """
    df = df.copy()
    clean_cols = []
    for col in df.columns:
        # strip bracketed and parenthesized text, units and punctuation
        c = re.sub(r'\[.*?\]|\(.*?\)|%|wt|ppm', '', col)
        c = re.sub(r'[\s\-\_]', '', c)  # remove spaces, hyphens, underscores
        c = c.lower()
        clean_cols.append(c)
    df.columns = clean_cols

    # now map those cleaned names to the canonical names used downstream
    mapping = {
        # major oxides
        'sio2':       'SiO2',
        'tio2':       'Ti',
        'al2o3':      'Al',
        'fe2o3t':     'Fe2O3',
        'feo':        'FeO',
        'mgo':        'MgO',
        'cao':        'CaO',
        'na2o':       'Na2O',
        'k2o':        'K2O',
        'p2o5':       'P2O5',
        # trace elements
        'th':         'Th',
        'nb':         'Nb',
        'zr':         'Zr',
        'y':          'Y',
        'sr':         'Sr',
        'la':         'La',
        'ce':         'Ce',
        'pr':         'Pr',
        'nd':         'Nd',
        'sm':         'Sm',
        'eu':         'Eu',
        'gd':         'Gd',
        'tb':         'Tb',
        'dy':         'Dy',
        'ho':         'Ho',
        'er':         'Er',
        'tm':         'Tm',
        'yb':         'Yb',
        'lu':         'Lu',
        'v':          'V',
        'sc':         'Sc',
        'co':         'Co',
        'ni':         'Ni',
        'cr':         'Cr',
        'hf':         'Hf',
        'ta':         'Ta',
        # isotopes
        '143nd/144nd':'143Nd/144Nd',
        '176hf/177hf':'176Hf/177Hf',
        '87sr/86sr':  '87Sr/86Sr',
        '206pb/204pb':'206Pb/204Pb',
        '207pb/204pb':'207Pb/204Pb'
    }
    df.rename(columns=mapping, inplace=True)
    return df

# -----------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------

def load_data(source):
    """Load a CSV (either file-like or StringIO) and sanitize its columns."""
    df = pd.read_csv(source)
    df = sanitize_columns(df)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only samples with 45 ≤ SiO₂ ≤ 57, if SiO₂ exists."""
    if 'SiO2' not in df.columns:
        return df
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute all custom ratios, anomalies, and ε-isotopes."""
    eps = 1e-6
    def add_ratio(name, num, den):
        if num in df.columns and den in df.columns:
            df[name] = df[num] / (df[den] + eps)

    # core ratios
    add_ratio('Th/La','Th','La')
    if all(c in df.columns for c in ['Ce','La','Pr']):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La']*df['Pr']) + eps)
    if all(c in df.columns for c in ['Eu','Sm','Gd']):
        df['Eu/Eu*'] = df['Eu'] / (np.sqrt(df['Sm']*df['Gd']) + eps)
    if all(c in df.columns for c in ['Nb','Zr','Ta']):
        df['Nb/Nb*'] = df['Nb'] / (np.sqrt(df['Zr']*df['Ta']) + eps)
    add_ratio('Nb/Zr','Nb','Zr')
    add_ratio('La/Zr','La','Zr')
    add_ratio('La/Yb','La','Yb')
    add_ratio('Sr/Y','Sr','Y')
    add_ratio('Gd/Yb','Gd','Yb')
    if all(c in df.columns for c in ['Nd','Pr','Sm']):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr']*df['Sm']) + eps)
    add_ratio('Dy/Yb','Dy','Yb')
    add_ratio('Th/Yb','Th','Yb')
    add_ratio('Nb/Yb','Nb','Yb')
    add_ratio('Zr/Y','Zr','Y')
    add_ratio('Ti/Zr','Ti','Zr')
    add_ratio('Y/Nb','Y','Nb')
    add_ratio('Th/Nb','Th','Nb')
    add_ratio('Ti/V','Ti','V')
    add_ratio('Ti/Al','Ti','Al')

    # ε‐isotopes
    if '143Nd/144Nd' in df.columns:
        CHUR = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR)/CHUR)*1e4
    if '176Hf/177Hf' in df.columns:
        CHUR = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR)/CHUR)*1e4

    return df

def preprocess_list(parts):
    """Concatenate, filter to basalt(+andesite), then compute all ratios."""
    full = pd.concat(parts, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")
st.markdown("Load GEOROC parts from GitHub **or** upload your own CSV parts.")

use_github     = st.checkbox("📦 Load GEOROC parts from GitHub")
uploaded_parts = st.file_uploader("Upload split GEOROC CSVs", type="csv", accept_multiple_files=True)

df_parts, skipped = [], []

# 1) GitHub path
if use_github:
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    # adjust filenames to match whatever you’ve split (e.g. 2024-12-2JETOA_tholeiitic_basalt_part1.csv, etc)
    filenames = [
        "2024-12-2JETOA_THOLEIITIC_BASALT_part1.csv",
        "2024-12-2JETOA_THOLEIITIC_BASALT_part2.csv",
        # … add all your parts here …
    ]
    for fn in filenames:
        try:
            r = requests.get(base + fn)
            dfp = load_data(StringIO(r.text))
            if dfp.isnull().any().any():
                skipped.append(fn)
            else:
                df_parts.append(dfp)
        except:
            skipped.append(fn)

    if not df_parts:
        st.error("❌ No valid GEOROC parts could be loaded from GitHub.")
        st.stop()

    df = preprocess_list(df_parts)
    if skipped:
        st.warning("⚠ Skipped: " + ", ".join(skipped))

# 2) Upload path
elif uploaded_parts:
    for up in uploaded_parts:
        try:
            dfp = load_data(up)
            if dfp.isnull().any().any():
                skipped.append(up.name)
            else:
                df_parts.append(dfp)
        except:
            skipped.append(up.name)

    if not df_parts:
        st.error("❌ All uploaded files were invalid or contained missing data.")
        st.stop()

    df = preprocess_list(df_parts)
    if skipped:
        st.warning("⚠ Skipped uploads: " + ", ".join(skipped))

# 3) neither chosen
else:
    st.info("▶️ Please select GitHub or upload CSVs to begin.")
    st.stop()

# -----------------------------
# PLOTS
# -----------------------------
if {'εNd','εHf'}.issubset(df.columns):
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set(xlabel="εNd", ylabel="εHf", title="εNd vs εHf")
    st.pyplot(fig)

if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e]/CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, normed, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set(title="REE Spider Plot")
    st.pyplot(fig2)

# -----------------------------
# COMPUTED RATIOS SUMMARY
# -----------------------------
st.subheader("Computed Ratios Summary")
ratio_cols = [
    'Th/La','Ce/Ce*','Eu/Eu*','Nb/Nb*',
    'Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb',
    'Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb',
    'Ti/Zr','Ti/Al','Y/Nb','Th/Nb','Ti/V'
]
available = [c for c in ratio_cols if c in df.columns]
if available:
    st.dataframe(df[available].describe().T.style.format("{:.2f}"))
else:
    st.warning("No computed ratios found.\nColumns present:\n" + ", ".join(df.columns))

# -----------------------------
# MACHINE-LEARNING CLASSIFICATION
# -----------------------------
st.subheader("Machine-Learning Classification (Tectonic Setting)")
labels  = [c for c in df.columns if df[c].nunique()<20]
defaults = MAJOR_OXIDES + REE_ELEMENTS + ['Ce/Ce*','Eu/Eu*','Nb/Nb*']

label_col = st.selectbox("Label column", labels)
train_features = st.multiselect(
    "Feature columns",
    options=[c for c in df.select_dtypes("number").columns],
    default=[f for f in defaults if f in df.columns]
)

if st.button("Run Classification") and train_features:
    dfc = df.dropna(subset=train_features+[label_col])
    X  = dfc[train_features]
    y  = dfc[label_col].astype('category').cat.codes
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.3,random_state=42)
    model = MLPClassifier((50,25),max_iter=500,early_stopping=True)
    model.fit(Xtr,ytr)
    preds = model.predict(Xte)
    st.text(classification_report(yte,preds,zero_division=0))

# -----------------------------
# DOWNLOAD PROCESSED DATA
# -----------------------------
st.subheader("Download Processed Data")
csv_bytes = df.to_csv(index=False).encode()
st.download_button("📥 Download CSV", data=csv_bytes,
                   file_name="geochem_results.csv",
                   mime="text/csv")
