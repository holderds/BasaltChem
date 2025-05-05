# basalt_load_and_clean.py

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Basalt Load & Clean", layout="wide")
st.title("Basalt Training Data: Load & Clean")

# 1) List of all CSVs in the repo with prefix "2024-12-2JETOA_"
BASE_URL = "https://raw.githubusercontent.com/holderds/basaltchem/main/"
CSV_FILES = [
    "2024-12-2JETOA_ALKALI_BASALT.csv",
    "2024-12-2JETOA_BASALTIC-ANDESITE_part1.csv",
    "2024-12-2JETOA_BASALTIC-ANDESITE_part2.csv",
    "2024-12-2JETOA_BASALTIC-TRACHYANDESITE.csv",
    "2024-12-2JETOA_BASALT_part1.csv",
    "2024-12-2JETOA_BASALT_part2.csv",
    "2024-12-2JETOA_BASALT_part3.csv",
    "2024-12-2JETOA_BASALT_part4.csv",
    "2024-12-2JETOA_BASALT_part5.csv",
    "2024-12-2JETOA_BASALT_part6.csv",
    "2024-12-2JETOA_BASALT_part7.csv",
    "2024-12-2JETOA_BASALT_part8.csv",
    "2024-12-2JETOA_BASALT_part9.csv",
    "2024-12-2JETOA_BASANITE.csv",
    "2024-12-2JETOA_BONINITE.csv",
    "2024-12-2JETOA_CALC-ALKALI_BASALT.csv",
    "2024-12-2JETOA_HAWAIITE.csv",
    "2024-12-2JETOA_MELILITITE.csv",
    "2024-12-2JETOA_MUGEARITE.csv",
    "2024-12-2JETOA_PICRITE.csv",
    "2024-12-2JETOA_SUBALKALI_BASALT.csv",
    "2024-12-2JETOA_THOLEIITIC_BASALT_part1.csv",
    "2024-12-2JETOA_THOLEIITIC_BASALT_part2.csv",
    "2024-12-2JETOA_TRACHYBASALT.csv",
    "2024-12-2JETOA_TRANSITIONAL_BASALT.csv",
]

# 2) Column rename map
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al',
    'Fe2O3(t) [wt%]': 'Fe2O3', 'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
    'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O', 'P2O5 [wt%]': 'P2O5',
    'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr', 'Y [ppm]': 'Y',
    'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb',
    'Dy [ppm]': 'Dy', 'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm',
    'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu', 'V [ppm]': 'V', 'Sc [ppm]': 'Sc',
    'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf'
}

# 3) Load & clean
dfs = []
errors = []
for fname in CSV_FILES:
    url = BASE_URL + fname
    try:
        # try utf-8
        df = pd.read_csv(url, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        # fallback to latin1
        df = pd.read_csv(url, low_memory=False, encoding="latin1")
    except Exception as e:
        errors.append(f"{fname}: {e}")
        continue

    # rename columns
    df = df.rename(columns=RENAME_MAP)

    # coerce to numeric (non-convertible → NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs.append(df)

# 4) Report load results
st.write(f"**Successfully loaded** {len(dfs)} of {len(CSV_FILES)} files.")
if errors:
    st.subheader("❌ Load Errors")
    for err in errors:
        st.write(f"- {err}")

if not dfs:
    st.stop()

# 5) Concatenate
df_all = pd.concat(dfs, ignore_index=True)

st.subheader("Combined DataFrame Preview")
st.write(f"Shape: {df_all.shape}")
st.dataframe(df_all.head(10))

st.subheader("Column List")
st.write(df_all.columns.tolist())
