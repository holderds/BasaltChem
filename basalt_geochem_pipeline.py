# basalt_geochem_pipeline.py

import requests, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# --- CONFIGURATION ---
GITHUB_REPO = "holderds/basaltchem"
BRANCH      = "main"
CSV_PREFIX  = "2024-12-2JETOA_"

# Map raw GEOROC headers → friendly names
RENAME_MAP = {
    'SiO2 [wt%]': 'SiO2', 'TiO2 [wt%]': 'Ti', 'Al2O3 [wt%]': 'Al',
    'Fe2O3(t) [wt%]': 'Fe2O3', 'MgO [wt%]': 'MgO', 'CaO [wt%]': 'CaO',
    'Na2O [wt%]': 'Na2O', 'K2O [wt%]': 'K2O', 'P2O5 [wt%]': 'P2O5',
    'Th [ppm]': 'Th', 'Nb [ppm]': 'Nb', 'Zr [ppm]': 'Zr', 'Y [ppm]': 'Y',
    'La [ppm]': 'La', 'Ce [ppm]': 'Ce', 'Pr [ppm]': 'Pr', 'Nd [ppm]': 'Nd',
    'Sm [ppm]': 'Sm', 'Eu [ppm]': 'Eu', 'Gd [ppm]': 'Gd', 'Tb [ppm]': 'Tb',
    'Dy [ppm]': 'Dy', 'Ho [ppm]': 'Ho', 'Er [ppm]': 'Er', 'Tm [ppm]': 'Tm',
    'Yb [ppm]': 'Yb', 'Lu [ppm]': 'Lu', 'V [ppm]': 'V', 'Sc [ppm]': 'Sc',
    'Co [ppm]': 'Co', 'Ni [ppm]': 'Ni', 'Cr [ppm]': 'Cr', 'Hf [ppm]': 'Hf',
}

# Chondrite values for REE
CHONDRITE = {
    'La':0.234,'Ce':0.624,'Pr':0.088,'Nd':0.492,'Sm':0.153,'Eu':0.060,
    'Gd':0.256,'Tb':0.040,'Dy':0.253,'Ho':0.058,'Er':0.165,'Tm':0.026,
    'Yb':0.161,'Lu':0.025
}

REE_ELEMENTS    = list(CHONDRITE)
MAJORS          = ['SiO2','Ti','Al','Fe2O3','MgO','CaO','Na2O','K2O','P2O5']
ANOMALIES       = ['Ce/Ce*','Eu/Eu*','Nb/Nb*']
ALL_FEATURES    = MAJORS + REE_ELEMENTS + ANOMALIES

# --- HELPERS ---

def list_csv_urls():
    api = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{BRANCH}?recursive=1"
    r   = requests.get(api)
    r.raise_for_status()
    tree = r.json().get('tree',[])
    return [
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{e['path']}"
        for e in tree
        if e['path'].startswith(CSV_PREFIX) and e['path'].endswith(".csv")
    ]

def load_clean(url):
    df = pd.read_csv(url, low_memory=False)
    df.rename(columns=RENAME_MAP, inplace=True)
    # coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop all-NaN cols
    return df.dropna(axis=1, how='all')

def filter_si(df):
    if 'SiO2' not in df.columns:
        st.error("❌ SiO2 column missing in data.")
        st.stop()
    return df[(df['SiO2']>=45)&(df['SiO2']<=57)]

def compute_ratios(df):
    eps = 1e-6
    # classic ratios
    df['Th/La']  = df['Th']  / (df['La']  + eps)
    df['Ce/Ce*'] = df['Ce']  / (np.sqrt(df['La']*df['Pr']) + eps)
    df['Eu/Eu*'] = df['Eu']  / (np.sqrt(df['Sm']*df['Gd']) + eps)
    df['Nb/Nb*'] = df['Nb']  / (np.sqrt(df['Zr']*df['Y'])  + eps)
    df['La/Yb']  = df['La']  / (df['Yb'] + eps)
    df['Dy/Yb']  = df['Dy']  / (df['Yb'] + eps)
    # Sr/Y if present
    if 'Sr' in df.columns:
        df['Sr/Y'] = df['Sr'] / (df['Y'] + eps)
    # isotopic ε‐values
    if {'143Nd/144Nd','176Hf/177Hf'}.issubset(df):
        CHUR_Nd,CHUR_Hf = 0.512638,0.282785
        df['εNd'] = ((df['143Nd/144Nd']-CHUR_Nd)/CHUR_Nd)*1e4
        df['εHf'] = ((df['176Hf/177Hf']-CHUR_Hf)/CHUR_Hf)*1e4
    # Pb isotope ratio
    if {'206Pb/204Pb','207Pb/204Pb'}.issubset(df):
        df['Pb_iso'] = df['206Pb/204Pb']/df['207Pb/204Pb']
    return df

# --- STREAMLIT APP ---

st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("🔄 Loading training CSVs from GitHub…")
urls = list_csv_urls()
if not urls:
    st.error("❌ No training CSVs found on GitHub.")
    st.stop()

parts, skipped = [], []
for u in urls:
    try:
        parts.append(load_clean(u))
    except Exception as e:
        skipped.append(u.split('/')[-1])

if not parts:
    st.error("❌ Could not load any valid CSVs.")
    st.stop()
elif skipped:
    st.warning(f"Skipped {len(skipped)} files: {', '.join(skipped)}")

# concatenate → filter → ratios
df = pd.concat(parts, ignore_index=True)
df = filter_si(df)
df = compute_ratios(df)

# — εNd vs εHf plot —
if {'εNd','εHf'}.issubset(df.columns):
    fig,ax = plt.subplots()
    ax.scatter(df['εNd'],df['εHf'],alpha=0.6)
    ax.set(xlabel="εNd",ylabel="εHf",title="εNd vs εHf")
    st.pyplot(fig)

# — REE spider —
if set(REE_ELEMENTS).issubset(df.columns):
    fig,ax = plt.subplots()
    for _,r in df.iterrows():
        norm = [r[e]/CHONDRITE[e] for e in REE_ELEMENTS]
        ax.plot(REE_ELEMENTS,norm,alpha=0.3)
    ax.set(yscale='log',ylabel='Chondrite-normalized',title='REE Spider')
    st.pyplot(fig)

# — Ratio summary —
ratio_list = [r for r in ANOMALIES+['Th/La','La/Yb','Dy/Yb','Sr/Y','Pb_iso'] if r in df]
if ratio_list:
    st.subheader("Computed Ratios")
    st.dataframe(df[ratio_list].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check element coverage.")

# — Classification —
st.subheader("Supervised Classification (Tectonic Setting)")
labels = [c for c in df.columns if df[c].nunique()<20]
label = st.selectbox("Label column", labels)
features = [c for c in ALL_FEATURES+ratio_list if c in df]
sel_feats = st.multiselect("Features", features, default=features[:10])

if sel_feats and label:
    sub = df.dropna(subset=sel_feats+[label])
    X,y = sub[sel_feats], sub[label].astype('category').cat.codes
    if len(X)>1:
        Xt, Xv, yt, yv = train_test_split(X,y,test_size=0.3,random_state=42)
        clf = MLPClassifier((50,25),max_iter=500,random_state=42).fit(Xt,yt)
        acc = np.mean(clf.predict(Xv)==yv)
        st.write("Accuracy:", f"{acc:.2%}")
    else:
        st.warning("Not enough complete rows for classification.")

# — Download cleaned data —
if st.button("Download cleaned CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download", data=csv, file_name="basaltchem_cleaned.csv", mime="text/csv")
