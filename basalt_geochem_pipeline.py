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
# Constants & Element Lists
# -----------------------------
CHONDRITE_VALUES = {
    'La':0.234,'Ce':1.59,'Pr':0.456,'Nd':1.23,'Sm':0.298,'Eu':0.123,
    'Gd':0.27,'Tb':0.1,'Dy':0.26,'Ho':0.059,'Er':0.167,'Tm':0.025,
    'Yb':0.17,'Lu':0.036
}
REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

# -----------------------------
# Helper Functions
# -----------------------------
def load_data(source):
    """Read & rename GEOROC CSV columns to standard names."""
    df = pd.read_csv(source)
    df = df.rename(columns={
        'SiO2 [wt%]':'SiO2','TiO2 [wt%]':'Ti','Al2O3 [wt%]':'Al',
        'Fe2O3(t) [wt%]':'Fe2O3','MgO [wt%]':'MgO','CaO [wt%]':'CaO',
        'Na2O [wt%]':'Na2O','K2O [wt%]':'K2O','P2O5 [wt%]':'P2O5',
        'Th [ppm]':'Th','Nb [ppm]':'Nb','Zr [ppm]':'Zr','Y [ppm]':'Y',
        'La [ppm]':'La','Ce [ppm]':'Ce','Pr [ppm]':'Pr','Nd [ppm]':'Nd',
        'Sm [ppm]':'Sm','Eu [ppm]':'Eu','Gd [ppm]':'Gd','Tb [ppm]':'Tb',
        'Dy [ppm]':'Dy','Ho [ppm]':'Ho','Er [ppm]':'Er','Tm [ppm]':'Tm',
        'Yb [ppm]':'Yb','Lu [ppm]':'Lu','V [ppm]':'V','Sc [ppm]':'Sc',
        'Co [ppm]':'Co','Ni [ppm]':'Ni','Cr [ppm]':'Cr','Hf [ppm]':'Hf',
        'Sr [ppm]':'Sr'
    })
    return df

def filter_basalt_to_basaltic_andesite(df):
    """Keep only 45–57 wt% SiO2."""
    return df[(df['SiO2']>=45)&(df['SiO2']<=57)]

def compute_ratios(df):
    """Add all your custom ratio columns."""
    eps = 1e-6
    df['Th/La']  = df['Th']/(df['La']+eps)
    df['Ce/Ce*']= df['Ce']/(np.sqrt(df['La']*df['Pr'])+eps)
    df['Nb/Zr'] = df['Nb']/(df['Zr']+eps)
    df['La/Zr'] = df['La']/(df['Zr']+eps)
    df['La/Yb'] = df['La']/(df['Yb']+eps)
    df['Sr/Y']  = df['Sr']/(df['Y']+eps)
    df['Gd/Yb'] = df['Gd']/(df['Yb']+eps)
    df['Nd/Nd*']= df['Nd']/(np.sqrt(df['Pr']*df['Sm'])+eps)
    df['Dy/Yb'] = df['Dy']/(df['Yb']+eps)
    df['Th/Yb'] = df['Th']/(df['Yb']+eps)
    df['Nb/Yb'] = df['Nb']/(df['Yb']+eps)
    df['Zr/Y']  = df['Zr']/(df['Y']+eps)
    df['Ti/Zr'] = df['Ti']/(df['Zr']+eps)
    df['Y/Nb']  = df['Y']/(df['Nb']+eps)
    df['Th/Nb'] = df['Th']/(df['Nb']+eps)
    if 'Ti' in df and 'V' in df: df['Ti/V']=df['Ti']/(df['V']+eps)
    if 'Ti' in df and 'Al' in df:df['Ti/Al']=df['Ti']/(df['Al']+eps)

    # εNd and εHf if isotopes present:
    if '143Nd/144Nd' in df:
        CHUR=0.512638
        df['εNd'] = ((df['143Nd/144Nd']-CHUR)/CHUR)*1e4
    if '176Hf/177Hf' in df:
        CHUR=0.282785
        df['εHf'] = ((df['176Hf/177Hf']-CHUR)/CHUR)*1e4

    return df

def preprocess_list(df_list):
    """Concat + filter + ratio-compute."""
    full = pd.concat(df_list, ignore_index=True)
    full = filter_basalt_to_basaltic_andesite(full)
    full = compute_ratios(full)
    return full

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("Basalt & Basaltic-Andesite Geochemistry Interpreter")

st.markdown("**Load** GEOROC parts from GitHub **or** **Upload** your own CSVs:")
use_github = st.checkbox("Load parts from GitHub")
uploads    = st.file_uploader("Upload CSV parts", type="csv", accept_multiple_files=True)

df_list, skipped = [], []

if use_github:
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    parts = [f"2024-12-2JETOA_part{i}.csv" for i in range(1,11)]
    for p in parts:
        try:
            resp = requests.get(base+p)
            part = load_data(StringIO(resp.text))
            if part.isnull().any().any(): skipped.append(p); continue
            df_list.append(part)
        except:
            skipped.append(p)
    if not df_list:
        st.error("No valid GEOROC parts loaded.")
        st.stop()
    df = preprocess_list(df_list)
    if skipped: st.warning("Skipped: "+" ,".join(skipped))

elif uploads:
    for f in uploads:
        try:
            part = load_data(f)
            if part.isnull().any().any(): skipped.append(f.name); continue
            df_list.append(part)
        except:
            skipped.append(f.name)
    if not df_list:
        st.error("All uploads invalid.")
        st.stop()
    df = preprocess_list(df_list)
    if skipped: st.warning("Skipped uploads: "+",".join(skipped))

else:
    st.info("Select GitHub or upload files.")
    st.stop()

# -----------------------------
# Plots
# -----------------------------
if 'εNd' in df and 'εHf' in df:
    fig,ax=plt.subplots(); ax.scatter(df['εNd'],df['εHf'],alpha=0.6)
    ax.set(xlabel="εNd",ylabel="εHf",title="εNd vs εHf"); st.pyplot(fig)

if all(e in df for e in REE_ELEMENTS):
    fig2,ax2=plt.subplots()
    for _,r in df.iterrows():
        norm=[r[e]/CHONDRITE_VALUES[e] for e in REE_ELEMENTS]
        ax2.plot(REE_ELEMENTS,norm,alpha=0.4)
    ax2.set_yscale('log'); ax2.set_title("REE Spider Plot")
    st.pyplot(fig2)

st.subheader("Ratio Summary")
ratios=['Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Nd/Nd*','Dy/Yb','Th/Yb','Nb/Yb']
available=[c for c in ratios if c in df]
st.write(df[available].describe().T.round(2))

# -----------------------------
# ML Classification
# -----------------------------
st.subheader("ML: Tectonic Setting")
label = st.selectbox("Label column", [c for c in df if df[c].nunique()<20])
features = st.multiselect("Features", df.select_dtypes("number").columns, default=available)

if st.button("Run ML") and features:
    subset=df.dropna(subset=features+[label])
    X=subset[features]; y=subset[label].astype('category').cat.codes
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42)
    mdl=MLPClassifier((50,25),max_iter=500,early_stopping=True).fit(Xtr,ytr)
    pr=mdl.predict(Xte)
    st.text(classification_report(yte,pr,zero_division=0))

# -----------------------------
# Export
# -----------------------------
st.subheader("Download Results")
csv_bytes = df.to_csv(index=False).encode()
st.download_button("Download CSV",csv_bytes,"geochem_results.csv","text/csv")
