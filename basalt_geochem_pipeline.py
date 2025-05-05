# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import streamlit as st
import io

# -----------------------------
# Function Definitions
# -----------------------------

def load_data(upload):
    """Read a CSV and rename GEOROC column names to our short names."""
    df = pd.read_csv(upload)
    rename_map = {
        'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al', 'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO', 'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5', 'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr',
        'Y [ppm]': 'Y', 'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb', 'Dy [ppm]': 'Dy',
        'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm', 'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu',
        'V [ppm]': 'V', 'Sc [ppm]': 'Sc', 'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def filter_basaltic(df):
    """Keep only 45–57 wt% SiO2."""
    return df.loc[(df["SiO2"] >= 45) & (df["SiO2"] <= 57)]

def compute_ratios(df):
    """Compute all custom geochemical ratios and ε-values."""
    eps = 1e-6
    # example ratio computations...
    df['Th/La']  = df['Th'] / (df['La'] + eps)
    df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']  = df['Nb'] / (df['Zr'] + eps)
    # ... (rest of your ratios)
    return df

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Basalt and Basaltic Andesite Geochemistry Interpreter")

st.write("Or load the cleaned GEOROC training set from GitHub:")
use_zip = st.checkbox("Load GEOROC training set from GitHub CSVs")
uploaded_files = st.file_uploader("Or upload your own split GEOROC CSVs", type="csv", accept_multiple_files=True)

if use_zip:
    import requests
    from io import StringIO

    github_csv_urls = [
        "https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part1.csv",
        "https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part2.csv",
        # etc...
    ]
    df_list = []
    skipped = []
    for url in github_csv_urls:
        try:
            r = requests.get(url)
            # 1) rename columns
            df_part = load_data(StringIO(r.text))
            # 2) force numeric
            for c in df_part.columns:
                df_part[c] = pd.to_numeric(df_part[c], errors="coerce")
            # 3) drop rows missing SiO2
            df_part = df_part.dropna(subset=["SiO2"])
            if df_part.empty:
                skipped.append(url)
                continue
            df_list.append(df_part)
        except Exception:
            skipped.append(url)

    if not df_list:
        st.error("❌ Could not load any valid GEOROC parts.")
        st.stop()

    # concat + cleanup + filter + ratios
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["SiO2"])
    df = filter_basaltic(df)
    df = compute_ratios(df)

    if skipped:
        st.warning(f"⚠ Skipped {len(skipped)} URL(s) due to missing data or errors.")

elif uploaded_files:
    df_list = []
    skipped = []
    for f in uploaded_files:
        try:
            df_part = load_data(f)
            for c in df_part.columns:
                df_part[c] = pd.to_numeric(df_part[c], errors="coerce")
            df_part = df_part.dropna(subset=["SiO2"])
            if df_part.empty:
                skipped.append(f.name)
                continue
            df_list.append(df_part)
        except Exception:
            skipped.append(f.name)

    if not df_list:
        st.error("❌ Could not load any valid uploaded files.")
        st.stop()

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["SiO2"])
    df = filter_basaltic(df)
    df = compute_ratios(df)

    if skipped:
        st.warning(f"⚠ Skipped {len(skipped)} file(s): {', '.join(skipped)}")

else:
    st.warning("Please either select the GitHub option or upload CSV files.")
    st.stop()

# -----------------------------
# Now the rest of your plotting & ML code can assume `df` has a valid SiO2 column
# -----------------------------
st.write(f"✅ Loaded {len(df)} samples after filtering to 45–57 wt% SiO₂.")

# e.g. show a quick preview
st.dataframe(df.head())

# ...insert your εNd vs εHf plot, REE spider plot, ratio summaries, and ML classification here...
