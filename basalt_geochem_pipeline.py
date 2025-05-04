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
# CONSTANTS
# -----------------------------
CHONDRITE_VALUES = {
    'La':0.234,'Ce':1.59,'Pr':0.456,'Nd':1.23,'Sm':0.298,'Eu':0.123,
    'Gd':0.27,'Tb':0.10,'Dy':0.26,'Ho':0.059,'Er':0.167,'Tm':0.025,
    'Yb':0.17,'Lu':0.036
}
REE_ELEMENTS = list(CHONDRITE_VALUES.keys())
MAJOR_OXIDES = ['SiO2','Ti','Al','Fe2O3','MgO','CaO','Na2O','K2O','P2O5']
ANOMALIES    = ['Ce/Ce*','Eu/Eu*','Nb/Nb*']

# -----------------------------
# Helpers to clean & coerce
# -----------------------------
def sanitize_columns(df):
    df = df.copy()
    newcols = []
    for c in df.columns:
        x = re.sub(r'\[.*?\]|\(.*?\)|%|wt|ppm','',c)
        x = re.sub(r'[\s\-\_]+','',x).lower()
        newcols.append(x)
    df.columns = newcols
    # map to standardized names
    mapping = {
      'sio2':'SiO2','tio2':'Ti','al2o3':'Al','fe2o3t':'Fe2O3',
      'mgo':'MgO','cao':'CaO','na2o':'Na2O','k2o':'K2O','p2o5':'P2O5',
      'th':'Th','nb':'Nb','zr':'Zr','y':'Y','sr':'Sr','la':'La','ce':'Ce',
      'pr':'Pr','nd':'Nd','sm':'Sm','eu':'Eu','gd':'Gd','tb':'Tb','dy':'Dy',
      'ho':'Ho','er':'Er','tm':'Tm','yb':'Yb','lu':'Lu','v':'V','sc':'Sc',
      'co':'Co','ni':'Ni','cr':'Cr','hf':'Hf','ta':'Ta',
      '143nd/144nd':'143Nd/144Nd','176hf/177hf':'176Hf/177Hf',
      '87sr/86sr':'87Sr/86Sr','206pb/204pb':'206Pb/204Pb','207pb/204pb':'207Pb/204Pb'
    }
    return df.rename(columns=mapping)

def load_data(source):
    # source can be a filename or file‐like
    df = pd.read_csv(source, low_memory=False)
    df = sanitize_columns(df)
    # force everything to numeric; non‐parsable → NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# -----------------------------
# Ratio computation & filtering
# -----------------------------
def filter_si(df):
    if 'SiO2' not in df.columns:
        return df.iloc[0:0]
    return df[(df['SiO2']>=45)&(df['SiO2']<=57)]

def compute_ratios(df):
    eps=1e-6
    R=lambda n,d,name: df.__setitem__(name, df[n]/(df[d]+eps)) if n in df and d in df else None

    # classic ratios
    R('Th','La','Th/La')
    if all(e in df for e in ('Ce','La','Pr')): df['Ce/Ce*']=df['Ce']/np.sqrt(df['La']*df['Pr']+eps)
    if all(e in df for e in ('Eu','Sm','Gd')): df['Eu/Eu*']=df['Eu']/np.sqrt(df['Sm']*df['Gd']+eps)
    if all(e in df for e in ('Nb','Zr','Ta')): df['Nb/Nb*']=df['Nb']/np.sqrt(df['Zr']*df['Ta']+eps)

    for n,d in [
      ('Nb','Zr'),('La','Zr'),('La','Yb'),
      ('Sr','Y'),('Gd','Yb'),('Dy','Yb'),
      ('Th','Yb'),('Nb','Yb'),('Zr','Y'),
      ('Ti','Zr'),('Ti','Al'),('Y','Nb'),
      ('Th','Nb'),('Ti','V')
    ]:
        R(n,d,f"{n}/{d}")

    # isotopic eps
    if '143Nd/144Nd' in df:
        CH=0.512638; df['εNd']=(df['143Nd/144Nd']-CH)/CH*1e4
    if '176Hf/177Hf' in df:
        CH=0.282785; df['εHf']=(df['176Hf/177Hf']-CH)/CH*1e4

    return df

def preprocess(parts):
    full = pd.concat(parts, ignore_index=True)
    full = filter_si(full)
    return compute_ratios(full)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Basalt & Andesite Geochemistry Interpreter")

use_gh     = st.checkbox("Load GEOROC CSVs from GitHub")
uploaded   = st.file_uploader("…or upload your CSV parts", type="csv", accept_multiple_files=True)

parts=[]
skipped=[]

if use_gh:
    base = "https://raw.githubusercontent.com/holderds/BasaltChem/main/"
    gh_files = [
      "2024-12-2JETOA_THOLEIITIC_BASALT_part1.csv",
      "2024-12-2JETOA_THOLEIITIC_BASALT_part2.csv",
      # add all your split filenames…
    ]
    for fn in gh_files:
        try:
            txt = requests.get(base+fn).text
            dfp = load_data(StringIO(txt))
            dfp = dfp.dropna(subset=['SiO2'])
            if dfp.empty: skipped.append(fn)
            else:          parts.append(dfp)
        except:
            skipped.append(fn)

elif uploaded:
    for f in uploaded:
        dfp = load_data(f)
        dfp = dfp.dropna(subset=['SiO2'])
        if dfp.empty: skipped.append(f.name)
        else:          parts.append(dfp)

else:
    st.info("☞ Please select GitHub or upload CSVs to begin.")
    st.stop()

if not parts:
    st.error("❌ No valid data loaded—check your file(s).")
    st.stop()

df = preprocess(parts)
if skipped:
    st.warning("⚠ Skipped: " + ", ".join(skipped))

# εNd vs εHf
if {'εNd','εHf'}.issubset(df.columns):
    fig,ax=plt.subplots()
    ax.scatter(df['εNd'],df['εHf'],alpha=0.6)
    ax.set(xlabel="εNd",ylabel="εHf",title="εNd vs εHf")
    st.pyplot(fig)

# REE spider
if all(e in df.columns for e in REE_ELEMENTS):
    fig,ax=plt.subplots()
    for _,r in df.iterrows():
        ax.plot(REE_ELEMENTS, [r[e]/CHONDRITE_VALUES[e] for e in REE_ELEMENTS],alpha=0.3)
    ax.set(yscale='log',title="REE Spider Plot")
    st.pyplot(fig)

# Ratio summary
st.subheader("Computed Ratios & Oxides")
all_cols = MAJOR_OXIDES + REE_ELEMENTS + ANOMALIES + [
    'Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb','Dy/Yb',
    'Th/Yb','Nb/Yb','Zr/Y','Ti/Zr','Ti/Al','Y/Nb',
    'Th/Nb','Ti/V','εNd','εHf'
]
avail = [c for c in all_cols if c in df.columns]
if avail:
    st.dataframe(df[avail].describe().T.style.format("{:.2f}"))
else:
    st.warning(
      "No ratios computed—your data must have at least one of: " +
      ", ".join(set(MAJOR_OXIDES+REE_ELEMENTS))
    )

# ML classification
st.subheader("ML Classification (Tectonic Setting)")
labels = [c for c in df.columns if df[c].nunique()<20]
defaults = [c for c in all_cols if c in df.columns]

label_col = st.selectbox("Label column", labels, index=0)
features  = st.multiselect("Features", df.select_dtypes(float).columns, default=defaults)

if st.button("Run Classification") and features:
    dfc = df.dropna(subset=features+[label_col])
    X,y = dfc[features], dfc[label_col].astype('category').cat.codes
    xtr,xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42)
    model = MLPClassifier((50,25),max_iter=500,early_stopping=True,random_state=42)
    model.fit(xtr,ytr)
    st.text(classification_report(yte, model.predict(xte), zero_division=0))

# Download
st.subheader("Download Processed Data")
st.download_button("📥 CSV", df.to_csv(index=False).encode(),
                   file_name="processed_geochem.csv", mime="text/csv")
