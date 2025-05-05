# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Column cleanup + renaming
# -----------------------------
# Map raw cleaned names → canonical short names
RENAME_MAP = {
    'SiO2': 'SiO2', 'TiO2': 'Ti', 'Al2O3': 'Al', 'Fe2O3': 'Fe2O3',
    'MgO': 'MgO', 'CaO': 'CaO', 'Na2O': 'Na2O', 'K2O': 'K2O',
    'P2O5': 'P2O5',
    # trace elements
    'Th': 'Th', 'Nb': 'Nb', 'Zr': 'Zr', 'Y': 'Y', 'La': 'La', 'Ce': 'Ce',
    'Pr': 'Pr', 'Nd': 'Nd', 'Sm': 'Sm', 'Eu': 'Eu', 'Gd': 'Gd',
    'Tb': 'Tb', 'Dy': 'Dy', 'Ho': 'Ho', 'Er': 'Er', 'Tm': 'Tm',
    'Yb': 'Yb', 'Lu': 'Lu', 'V': 'V', 'Sc': 'Sc', 'Co': 'Co',
    'Ni': 'Ni', 'Cr': 'Cr', 'Hf': 'Hf'
}

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip any ' [units]' from every column name and trim whitespace."""
    cols = (
        df.columns
          .str.strip()
          .str.replace(r"\s*\[.*?\]", "", regex=True)
    )
    df = df.copy()
    df.columns = cols
    return df

def load_and_clean(src) -> pd.DataFrame:
    """Read CSV, clean names, rename to canonical, coerce key columns to numeric."""
    df = pd.read_csv(src, low_memory=False)
    df = clean_columns(df)
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce all canonical columns to numeric
    for canonical in RENAME_MAP.values():
        if canonical in df.columns:
            df[canonical] = pd.to_numeric(df[canonical], errors="coerce")
    return df

# -----------------------------
# Geochem processing
# -----------------------------
def filter_basaltic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 45–57 wt% SiO2."""
    return df.loc[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)]

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    # major‐trace
    df["Th/La"]  = df["Th"]  / (df["La"]  + eps)
    df["Ce/Ce*"] = df["Ce"]  / (np.sqrt(df["La"] * df["Pr"]) + eps)
    df["Nb/Zr"]  = df["Nb"]  / (df["Zr"]  + eps)
    df["La/Zr"]  = df["La"]  / (df["Zr"]  + eps)
    df["La/Yb"]  = df["La"]  / (df["Yb"]  + eps)
    df["Sr/Y"]   = df["Sr"]  / (df["Y"]   + eps)
    df["Gd/Yb"]  = df["Gd"]  / (df["Yb"]  + eps)
    df["Nd/Nd*"] = df["Nd"]  / (np.sqrt(df["Pr"] * df["Sm"]) + eps)
    df["Dy/Yb"]  = df["Dy"]  / (df["Yb"]  + eps)
    df["Th/Yb"]  = df["Th"]  / (df["Yb"]  + eps)
    df["Nb/Yb"]  = df["Nb"]  / (df["Yb"]  + eps)
    df["Zr/Y"]   = df["Zr"]  / (df["Y"]   + eps)
    df["Ti/Zr"]  = df["Ti"]  / (df["Zr"]  + eps)
    df["Y/Nb"]   = df["Y"]   / (df["Nb"]  + eps)
    df["Th/Nb"]  = df["Th"]  / (df["Nb"]  + eps)
    if {"Ti","V"}.issubset(df.columns):
        df["Ti/V"] = df["Ti"] / (df["V"] + eps)
    if {"Ti","Al"}.issubset(df.columns):
        df["Ti/Al"] = df["Ti"] / (df["Al"] + eps)
    # indices
    if "Dy/Yb" in df.columns and "La/Yb" in df.columns:
        df["Crustal_Thickness_Index"] = df["Dy/Yb"]
        df["Melting_Depth_Index"]      = df["La/Yb"]
    if "Th/Nb" in df.columns:
        df["Crustal_Contamination_Index"] = df["Th/Nb"]
    if "Nb/Y" in df.columns:
        df["Mantle_Fertility_Index"]      = df["Nb/Y"]
    if {"Sm","Nd"}.issubset(df.columns):
        df["Sm/Nd"] = df["Sm"] / (df["Nd"] + eps)
    return df

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("Load your split GEOROC CSV parts (either from GitHub or upload):")

use_github = st.checkbox("⬇️ Load cleaned GEOROC parts from GitHub")
uploads    = st.file_uploader(
    "📂 Or upload your own split parts", 
    type="csv", 
    accept_multiple_files=True
)

parts, skipped = [], []

if use_github:
    import requests
    for i in range(1,6):
        url = (
          "https://raw.githubusercontent.com/"
          "holderds/basaltchem/main/example_data/"
          f"cleaned_georoc_basalt_part{i}.csv"
        )
        try:
            txt  = requests.get(url).text
            part = load_and_clean(io.StringIO(txt))
            if part.isnull().any().any():
                skipped.append(f"part{i}")
            else:
                parts.append(part)
        except:
            skipped.append(f"part{i}")

elif uploads:
    for up in uploads:
        try:
            part = load_and_clean(up)
            if part.isnull().any().any():
                skipped.append(up.name)
            else:
                parts.append(part)
        except:
            skipped.append(up.name)

else:
    st.warning("📥 Please tick 'Load from GitHub' or upload at least one CSV.")
    st.stop()

if not parts:
    st.error(f"❌ Could not load any valid parts. Skipped: {skipped}")
    st.stop()

# concat → filter → ratios
df = pd.concat(parts, ignore_index=True)
df = filter_basaltic(df)
df = compute_ratios(df)

if skipped:
    st.warning(f"⚠️ Skipped files: {', '.join(skipped)}")

# -----------------------------
# REE Spider Plot
# -----------------------------
CHONDRITE = {
  'La':0.234,'Ce':1.59,'Pr':0.36,'Nd':1.23,'Sm':0.24,
  'Eu':0.12,'Gd':0.27,'Tb':0.04,'Dy':0.26,'Ho':0.06,
  'Er':0.17,'Tm':0.03,'Yb':0.17,'Lu':0.03
}
REE = list(CHONDRITE.keys())
if all(e in df.columns for e in REE):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        ax.plot(REE, [row[e]/CHONDRITE[e] for e in REE], alpha=0.3)
    ax.set_yscale('log')
    ax.set_title("REE Spider Diagram")
    ax.set_ylabel("Chondrite‐normalized")
    st.pyplot(fig)

# -----------------------------
# Ratios Summary
# -----------------------------
ratio_cols = [c for c in df.columns if "/" in c and not c.startswith(("143","176","87","206","207","ε"))]
if ratio_cols:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check your data.")

# -----------------------------
# Download
# -----------------------------
if st.button("Download Processed Data as CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "processed_georoc.csv", "text/csv")

# -----------------------------
# ML Classification
# -----------------------------
st.subheader("ML Classification: Tectonic Setting")

# exclude raw isotopes & ε‐values
exclude = {
    "143Nd/144Nd","176Hf/177Hf","87Sr/86Sr",
    "206Pb/204Pb","207Pb/204Pb","εNd","εHf"
}
numeric = df.select_dtypes("number").columns
features = [c for c in numeric if c not in exclude]

label_col  = st.selectbox("Label Column",
                         [c for c in df.columns if df[c].nunique()<20])
train_feats= st.multiselect("Features", features, default=features[:10])

if label_col and train_feats:
    dfc = df.dropna(subset=[label_col] + train_feats)
    if len(dfc) < 2:
        st.error("Not enough samples for training/testing.")
    else:
        X_train,X_test,y_train,y_test = train_test_split(
            dfc[train_feats],
            dfc[label_col].astype("category").cat.codes,
            test_size=0.3, random_state=42
        )
        clf = MLPClassifier((50,25), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        proba  = clf.predict_proba(X_test)
        rep    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        dfp = pd.DataFrame(
            proba, 
            columns=pd.Categorical(dfc[label_col]).categories
        )
        dfp["True"]      = pd.Categorical(dfc[label_col]).categories[y_test]
        dfp["Predicted"] = pd.Categorical(dfc[label_col]).categories[y_pred]

        st.subheader("Prediction Probabilities")
        st.dataframe(dfp.head())
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(rep).transpose())
