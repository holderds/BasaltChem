# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import streamlit as st
import io
import requests
from io import StringIO

# -----------------------------
# Function Definitions
# -----------------------------

def load_data(src):
    """Read CSV and rename GEOROC columns into simple element names."""
    df = pd.read_csv(src, low_memory=False)
    rename = {
        'SiO2 [wt%]': 'SiO2',
        'TiO2 [wt%]': 'Ti',
        'Al2O3 [wt%]': 'Al',
        'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO',
        'CaO [wt%]': 'CaO',
        'Na2O [wt%]': 'Na2O',
        'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5',
        # trace & REE
        **{f'{el} [ppm]': el for el in [
            'Th','Nb','Zr','Y','La','Ce','Pr','Nd','Sm','Eu',
            'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','V','Sc','Co','Ni','Cr','Hf'
        ]},
        # add Sr!
        'Sr [ppm]': 'Sr',
        # isotopes
        '87Sr/86Sr': 'Sr87_86',
        '206Pb/204Pb': 'Pb206_204',
        '207Pb/204Pb': 'Pb207_204',
        '143Nd/144Nd': 'Nd143_144',
        '176Hf/177Hf': 'Hf176_177'
    }
    df.rename(columns=rename, inplace=True)
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only samples with 45–57 wt% SiO₂."""
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Compute custom geochemical ratios and ε-isotope values."""
    eps = 1e-6

    # Major‐trace ratios
    df['Th/La']  = df['Th']  / (df['La'] + eps)
    df['Ce/Ce*'] = df['Ce']  / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']  = df['Nb']  / (df['Zr'] + eps)
    df['La/Zr']  = df['La']  / (df['Zr'] + eps)
    df['La/Yb']  = df['La']  / (df['Yb'] + eps)
    df['Sr/Y']   = df['Sr']  / (df['Y']  + eps)
    df['Gd/Yb']  = df['Gd']  / (df['Yb'] + eps)
    df['Nd/Nd*'] = df['Nd']  / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb']  = df['Dy']  / (df['Yb'] + eps)
    df['Th/Yb']  = df['Th']  / (df['Yb'] + eps)
    df['Nb/Yb']  = df['Nb']  / (df['Yb'] + eps)
    df['Zr/Y']   = df['Zr']  / (df['Y']  + eps)
    if 'Ti' in df.columns and 'V' in df.columns:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Ti' in df.columns and 'Al' in df.columns:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)

    # Sm/Nd ratio
    df['Sm/Nd'] = df['Sm'] / (df['Nd'] + eps)

    # εNd, εHf
    if 'Nd143_144' in df.columns:
        CHUR_Nd = 0.512638
        df['εNd'] = ((df['Nd143_144'] - CHUR_Nd) / CHUR_Nd) * 1e4
    if 'Hf176_177' in df.columns:
        CHUR_Hf = 0.282785
        df['εHf'] = ((df['Hf176_177'] - CHUR_Hf) / CHUR_Hf) * 1e4

    # Composite indices
    df['Crustal_Thickness_Index']     = df['Dy/Yb']
    df['Melting_Depth_Index']         = df['La/Yb']
    df['Crustal_Contamination_Index'] = df['Th/Nb']
    df['Mantle_Fertility_Index']      = df['Nb/Zr']

    return df

def preprocess_list(df_list):
    """Concatenate parts, filter, and compute all ratios."""
    full = pd.concat(df_list, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt & Basaltic Andesite Geochem Interpreter")

st.write("Load GEOROC CSV parts or use example dataset:")
use_zip        = st.checkbox("Load GEOROC training set from GitHub CSVs")
uploaded_files = st.file_uploader("Upload split GEOROC parts", type=["csv"], accept_multiple_files=True)
use_example    = st.checkbox("Use cleaned example dataset")

df = None
skipped = []

if use_zip:
    urls = [
        f"https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part{i}.csv"
        for i in range(1,6)
    ]
    parts = []
    for u in urls:
        try:
            r = requests.get(u)
            part = pd.read_csv(StringIO(r.text), low_memory=False)
            parts.append(part)
        except Exception:
            skipped.append(u)
    if not parts:
        st.error("❌ No valid GEOROC parts loaded from GitHub.")
        st.stop()
    df = preprocess_list(parts)

elif uploaded_files:
    parts = []
    for f in uploaded_files:
        try:
            part = load_data(f)
            parts.append(part)
        except Exception:
            skipped.append(f.name)
    if not parts:
        st.error("❌ No valid uploads.")
        st.stop()
    df = preprocess_list(parts)

elif use_example:
    example_url = "https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part1.csv"
    r = requests.get(example_url)
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    df = filter_basalt_to_basaltic_andesite(df)
    df = compute_ratios(df)

else:
    st.warning("Please choose an input method above.")
    st.stop()

if skipped:
    st.warning("Skipped: " + ", ".join(skipped))

# -----------------------------
# εNd vs εHf Isotope Scatter
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd")
    ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf")
    st.pyplot(fig)
    st.session_state['last_fig'] = fig

# -----------------------------
# REE Spider Plot
REE_ELEMENTS = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
CHONDRITE = {'La':0.237,'Ce':1.59,'Pr':0.49,'Nd':1.23,'Sm':0.28,'Eu':0.11,'Gd':0.27,'Tb':0.04,'Dy':0.26,'Ho':0.05,'Er':0.17,'Tm':0.02,'Yb':0.17,'Lu':0.03}

if all(e in df.columns for e in REE_ELEMENTS):
    fig2, ax2 = plt.subplots()
    for _, row in df.iterrows():
        norm = [row[e]/CHONDRITE[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS, norm, alpha=0.4)
    ax2.set_yscale('log')
    ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)
    st.session_state['last_fig'] = fig2

# -----------------------------
# Summary of Ratios
ratio_cols = ['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb']
available = [c for c in ratio_cols if c in df.columns]
if available:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[available].describe().T.style.format("{:.2f}"))
else:
    st.error("No ratios computed—check that your data contain the required elements.")

# -----------------------------
# Download & Export
if st.button("Download All Results as CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download", data=csv, file_name="basalt_geochem_results.csv", mime="text/csv")

if 'last_fig' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_fig'].savefig(buf, format="png")
    st.download_button("📷 Download Last Plot", data=buf.getvalue(), file_name="last_plot.png", mime="image/png")

# -----------------------------
# ML Classification
st.subheader("Supervised Classification: Tectonic Setting")
label_col = st.selectbox("Label column", [c for c in df.columns if df[c].nunique() < 20])
features  = st.multiselect("Features", df.select_dtypes("number").columns.tolist())

if label_col and features:
    df_cl = df.dropna(subset=[label_col]+features)
    X = df_cl[features]
    y = df_cl[label_col].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = MLPClassifier((50,25), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    st.text("Classification Report:")
    st.text(report)

    # Attach predictions
    df['Predicted_Label'] = None
    df.loc[df_cl.index, 'Predicted_Label'] = model.predict(X)

