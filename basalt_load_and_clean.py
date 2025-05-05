# basalt_load_and_clean.py

import requests
import pandas as pd
import streamlit as st

# --- CONFIG ---
GITHUB_REPO = "holderds/basaltchem"
BRANCH      = "main"
PREFIX      = "2024-12-2JETOA_"

# --- HELPERS ---
def list_csv_urls():
    """Query GitHub API to list our CSVs."""
    api = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{BRANCH}?recursive=1"
    res = requests.get(api)
    res.raise_for_status()
    items = res.json().get("tree", [])
    return [
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{i['path']}"
        for i in items
        if i["path"].startswith(PREFIX) and i["path"].endswith(".csv")
    ]

def clean_df(df):
    """Rename columns, coerce to numeric, drop all-empty columns."""
    # example rename map — adjust as needed
    rename_map = {
        'SiO2 [wt%]': 'SiO2',
        'TiO2 [wt%]': 'Ti',
        # …add more if you like
    }
    df = df.rename(columns=rename_map)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(axis=1, how='all')

# --- APP ---
st.title("🔄 Load & Clean GEOROC CSVs")

urls = list_csv_urls()
st.write("Found CSVs:", urls)

if not urls:
    st.error("❌ No CSV files found.")
    st.stop()

dfs = []
skipped = []
for u in urls:
    try:
        df = pd.read_csv(u, low_memory=False)
        dfs.append(clean_df(df))
    except Exception as e:
        skipped.append(u.split("/")[-1])

if skipped:
    st.warning(f"Skipped loading: {', '.join(skipped)}")

if not dfs:
    st.error("❌ None of the CSVs could be loaded!")
    st.stop()

full = pd.concat(dfs, ignore_index=True)
st.success("✅ All files loaded & cleaned.")
st.subheader("Preview of concatenated DataFrame")
st.dataframe(full.head())

st.write("DataFrame info:")
st.text(str(full.info()))
