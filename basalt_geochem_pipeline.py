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

# -----------------------------
# Utility Functions
# -----------------------------

def load_data(source):
    """
    Load a CSV (either a file-like or a URL text buffer),
    rename GEOROC columns to short names, coerce to numeric.
    """
    df = pd.read_csv(source)
    # Standardize column names:
    rename_map = {
        'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5',
        'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr', 'Y [ppm]': 'Y',
        'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb',
        'Dy [ppm]': 'Dy', 'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm',
        'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
        'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni',
        'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
    }
    df.rename(columns=rename_map, inplace=True)

    # Coerce everything to numeric (bad entries → NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def filter_basaltic(df):
    """Keep only samples with 45–57 wt% SiO₂."""
    return df.loc[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)]

def compute_ratios(df):
    """Compute your custom ratios and ε-values."""
    eps = 1e-6
    # Major/trace ratios
    df['Th/La']   = df['Th'] / (df['La'] + eps)
    df['Ce/Ce*']  = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']   = df['Nb'] / (df['Zr'] + eps)
    df['La/Zr']   = df['La'] / (df['Zr'] + eps)
    df['La/Yb']   = df['La'] / (df['Yb'] + eps)
    df['Sr/Y']    = df['Sr'] / (df['Y'] + eps)
    df['Gd/Yb']   = df['Gd'] / (df['Yb'] + eps)
    df['Nd/Nd*']  = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb']   = df['Dy'] / (df['Yb'] + eps)
    df['Th/Yb']   = df['Th'] / (df['Yb'] + eps)
    df['Nb/Yb']   = df['Nb'] / (df['Yb'] + eps)
    df['Zr/Y']    = df['Zr'] / (df['Y'] + eps)
    if 'Ti' in df.columns and 'V' in df.columns:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Ti' in df.columns and 'Al' in df.columns:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)

    # ε-values (if isotopes are present)
    if '143Nd/144Nd' in df.columns:
        CHUR_Nd = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR_Nd) / CHUR_Nd) * 1e4
    if '176Hf/177Hf' in df.columns:
        CHUR_Hf = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR_Hf) / CHUR_Hf) * 1e4
    if '206Pb/204Pb' in df.columns and '207Pb/204Pb' in df.columns:
        df['Pb_iso'] = df['206Pb/204Pb'] / (df['207Pb/204Pb'] + eps)

    return df

def preprocess_list(sources, from_url=False):
    """
    Given a list of either file-like objects or URLs, 
    load_data → drop rows missing SiO2 → concatenate → filter → compute ratios.
    Returns (df, skipped_list).
    """
    df_parts = []
    skipped = []

    for src in sources:
        try:
            if from_url:
                resp = requests.get(src)
                data = StringIO(resp.text)
                part = load_data(data)
            else:
                part = load_data(src)

            # drop any row missing our key column
            part = part.dropna(subset=["SiO2"])
            if part.empty:
                skipped.append(src if not from_url else src)
                continue

            df_parts.append(part)

        except Exception:
            skipped.append(src)

    if not df_parts:
        return None, skipped

    df = pd.concat(df_parts, ignore_index=True)
    df = filter_basaltic(df)
    df = compute_ratios(df)
    return df, skipped

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("Either load the cleaned GEOROC parts from GitHub or upload your own CSVs:")
use_github = st.checkbox("▶ Load cleaned GEOROC parts from GitHub")
uploaded = st.file_uploader("Or upload split GEOROC CSV parts", type="csv", accept_multiple_files=True)

if use_github:
    urls = [
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt_part1.csv",
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt_part2.csv",
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt_part3.csv",
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt_part4.csv",
        "https://raw.githubusercontent.com/holderds/BasaltChem/main/example_data/cleaned_georoc_basalt_part5.csv",
    ]
    df, skipped = preprocess_list(urls, from_url=True)

    if df is None:
        st.error("❌ Could not load any valid GEOROC parts from GitHub.")
        st.stop()

    if skipped:
        st.warning(f"⚠ Skipped {len(skipped)} GitHub URL(s) due to errors or missing SiO₂.")

elif uploaded:
    df, skipped = preprocess_list(uploaded, from_url=False)

    if df is None:
        st.error("❌ Could not load any valid uploaded CSVs.")
        st.stop()

    if skipped:
        st.warning(f"⚠ Skipped {len(skipped)} uploaded file(s): {', '.join(getattr(f,'name',str(f)) for f in skipped)}")

else:
    st.warning("Please select **Load from GitHub** or upload at least one CSV.")
    st.stop()

# -----------------------------
# Data is now loaded into `df` with:
#   • SiO2 guaranteed numeric, filtered to 45–57 wt%
#   • All custom ratios & ε-values computed
# -----------------------------

st.success(f"✅ Loaded **{len(df)}** samples (45–57 wt% SiO₂) with all ratios computed.")
st.dataframe(df.head())

# — Insert your plotting, ratio summaries, and ML sections below —
