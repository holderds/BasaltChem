# basalt_geochem_pipeline.py

import re
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# -----------------------------
# Helper functions
# -----------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip units from column names, coerce numeric columns, drop empty columns.
    """
    # 1) Standardize column names: strip whitespace, remove anything in [brackets]
    clean_cols = []
    for col in df.columns:
        col0 = col.strip()
        # remove unit annotations like " [wt%]" or " (ppm)"
        col1 = re.sub(r"\s*\[.*?\]", "", col0)
        col1 = re.sub(r"\s*\(.*?\)", "", col1)
        clean_cols.append(col1)
    df.columns = clean_cols

    # 2) Convert all columns except obviously non-numeric to numeric (coerce errors to NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Drop any columns that are now all-NaN
    df.dropna(axis=1, how="all", inplace=True)

    return df

def load_and_clean_csv(source) -> pd.DataFrame:
    """
    Load a CSV (file-like or URL string), clean it, and return DataFrame.
    """
    if isinstance(source, str) and source.startswith("http"):
        text = requests.get(source).text
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    else:
        df = pd.read_csv(source, low_memory=False)
    return clean_dataframe(df)

def filter_basaltic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only samples with 45 ≤ SiO2 ≤ 57.
    """
    if "SiO2" not in df.columns:
        st.error("❌ SiO₂ column missing—check your data.")
        st.stop()
    return df.loc[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)]

def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all trace‐element and REE ratios (but skip isotopes for ML).
    """
    eps = 1e-6

    # Major‐trace ratios
    for num, den in [("Th","La"), ("Ce","Ce*"), ("Nb","Zr"), ("La","Zr"), ("La","Yb"),
                     ("Sr","Y"), ("Gd","Yb"), ("Nd","Nd*"), ("Dy","Yb"), ("Th","Nb"), ("Nb","Yb")]:
        if num in df.columns and den in df.columns:
            if den == "Ce*":
                # Ce* = sqrt(La*Pr)
                if "La" in df.columns and "Pr" in df.columns:
                    df["Ce/Ce*"] = df["Ce"] / (np.sqrt(df["La"] * df["Pr"]) + eps)
            elif den == "Nd*":
                if "Pr" in df.columns and "Sm" in df.columns:
                    df["Nd/Nd*"] = df["Nd"] / (np.sqrt(df["Pr"] * df["Sm"]) + eps)
            else:
                df[f"{num}/{den}"] = df[num] / (df[den] + eps)

    # Imobile‐element ratios
    for a,b in [("Ti","Zr"), ("Y","Nb")]:
        if a in df.columns and b in df.columns:
            df[f"{a}/{b}"] = df[a] / (df[b] + eps)

    # Specialized indices if both pieces exist
    if all(x in df.columns for x in ("La/Yb","Dy/Yb")):
        df["Crustal_Thickness_Index"] = df["Dy/Yb"]
        df["Melting_Depth_Index"]        = df["La/Yb"]
    if "Th/Nb" in df.columns:
        df["Crustal_Contamination_Index"] = df["Th/Nb"]
    if "Nb/Y" in df.columns:
        df["Mantle_Fertility_Index"] = df["Nb/Y"]

    return df

# -----------------------------
# Streamlit UI & Logic
# -----------------------------

st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

use_zip       = st.checkbox("Load GEOROC training set from GitHub")
uploads       = st.file_uploader("Or upload your split GEOROC CSV parts", type="csv",
                                 accept_multiple_files=True)
df_list, skipped = [], []

if use_zip:
    urls = [
        f"https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part{i}.csv"
        for i in range(1,6)
    ]
    for url in urls:
        try:
            df_part = load_and_clean_csv(url)
            if df_part.empty:
                skipped.append(url)
            else:
                df_list.append(df_part)
        except Exception:
            skipped.append(url)
elif uploads:
    for f in uploads:
        try:
            df_part = load_and_clean_csv(f)
            if df_part.empty:
                skipped.append(f.name)
            else:
                df_list.append(df_part)
        except Exception:
            skipped.append(f.name)
else:
    st.warning("Please either check ‘Load GEOROC…’ or upload CSV files.")
    st.stop()

if not df_list:
    st.error("❌ Could not load any valid GEOROC parts.")
    st.stop()
if skipped:
    st.warning(f"⚠ Skipped: {', '.join(skipped)}")

# Concatenate, filter, compute
df = pd.concat(df_list, ignore_index=True)
df = filter_basaltic(df)
df = compute_ratios(df)

# --- Plots ---
if {"εNd","εHf"}.issubset(df.columns):
    fig,ax = plt.subplots()
    ax.scatter(df["εNd"],df["εHf"],alpha=0.6)
    ax.set(xlabel="εNd", ylabel="εHf", title="εNd vs εHf")
    st.pyplot(fig)
    st.session_state["last_fig"] = fig

# REE spider
REE = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
if all(e in df.columns for e in REE):
    CHONDRITE = {'La':0.237,'Ce':1.5,'Pr':0.4,'Nd':0.5,'Sm':0.2,'Eu':0.1,'Gd':0.2,'Tb':0.05,
                 'Dy':0.2,'Ho':0.05,'Er':0.1,'Tm':0.02,'Yb':0.1,'Lu':0.02}
    fig2,ax2 = plt.subplots()
    for _,r in df.iterrows():
        normed = [r[e]/CHONDRITE[e] for e in REE]
        ax2.plot(REE,normed,alpha=0.3)
    ax2.set(yscale="log", ylabel="Chondrite‐normalized", title="REE Spider Plot")
    st.pyplot(fig2)
    st.session_state["last_fig"] = fig2

# Preview computed ratios
ratio_cols = [c for c in df.columns if "/" in c and c not in ("143Nd/144Nd","176Hf/177Hf",
                                                               "87Sr/86Sr","206Pb/204Pb","207Pb/204Pb")]
if ratio_cols:
    st.subheader("Computed Ratios")
    st.dataframe(df[ratio_cols].describe().T.style.format("{:.2f}"))

# --- Classification (excluding isotopes) ---
st.subheader("MLP Classification (Tectonic Setting)")
# label choices: any column with few unique values
labels = [c for c in df.columns if df[c].nunique() < 20]
label_col = st.selectbox("Label column", labels)

# features: numeric columns excluding isotope columns
iso_cols  = ["143Nd/144Nd","176Hf/177Hf","87Sr/86Sr","206Pb/204Pb","207Pb/204Pb",
             "εNd","εHf","Pb_iso"]
features = [c for c in df.select_dtypes("number").columns
            if c not in iso_cols + [label_col]]

selected = st.multiselect("Features", features, default=features[:10])
if selected and label_col:
    sub = df.dropna(subset=selected+[label_col])
    if len(sub) < 2:
        st.error("Not enough data for classification after dropping NaNs.")
    else:
        X = sub[selected]
        y = sub[label_col].astype("category").cat.codes
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        clf = MLPClassifier((50,25),max_iter=500,random_state=42)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        st.write(classification_report(y_test,y_pred,zero_division=0))
        # show probability table for first few
        proba = pd.DataFrame(clf.predict_proba(X_test),
                             columns=clf.classes_).assign(
                             True=y_test.map(dict(zip(range(len(clf.classes_)),clf.classes_))),
                             Pred=clf.classes_[y_pred])
        st.dataframe(proba.head())

# --- Downloads ---
if st.button("Download processed data"):
    btn = st.download_button("📥 CSV", data=df.to_csv(index=False),
                             file_name="processed_georoc.csv",
                             mime="text/csv")
if "last_fig" in st.session_state:
    buf = io.BytesIO()
    st.session_state["last_fig"].savefig(buf,format="png")
    st.download_button("📷 Download last plot", data=buf.getvalue(),
                       file_name="plot.png", mime="image/png")
