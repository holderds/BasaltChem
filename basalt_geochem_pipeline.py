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
# Functions
# -----------------------------

def load_data(source):
    """Read CSV and rename GEOROC-style columns to simple names."""
    df = pd.read_csv(source, low_memory=False)
    rename_map = {
        'SiO2 [wt%]': 'SiO2',
        'TiO2 [wt%]': 'Ti',
        'Al2O3 [wt%]': 'Al',
        'Fe2O3(t) [wt%]': 'Fe2O3',
        'MgO [wt%]': 'MgO',
        'CaO [wt%]': 'CaO',
        'Na2O [wt%]': 'Na2O',
        'K2O [wt%]': 'K2O',
        'P2O5 [wt%]': 'P2O5',
        'Th [ppm]': 'Th',
        'Nb [ppm]': 'Nb',
        'Zr [ppm]': 'Zr',
        'Y [ppm]': 'Y',
        'La [ppm]': 'La',
        'Ce [ppm]': 'Ce',
        'Pr [ppm]': 'Pr',
        'Nd [ppm]': 'Nd',
        'Sm [ppm]': 'Sm',
        'Eu [ppm]': 'Eu',
        'Gd [ppm]': 'Gd',
        'Tb [ppm]': 'Tb',
        'Dy [ppm]': 'Dy',
        'Ho [ppm]': 'Ho',
        'Er [ppm]': 'Er',
        'Tm [ppm]': 'Tm',
        'Yb [ppm]': 'Yb',
        'Lu [ppm]': 'Lu',
        'V [ppm]': 'V',
        'Sc [ppm]': 'Sc',
        'Co [ppm]': 'Co',
        'Ni [ppm]': 'Ni',
        'Cr [ppm]': 'Cr',
        'Hf [ppm]': 'Hf'
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def filter_basaltic(df):
    """Keep only 45–57 wt% SiO2."""
    return df.loc[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    df['Th/La']   = df['Th']  / (df['La']  + eps)
    df['Ce/Ce*']  = df['Ce']  / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr']   = df['Nb']  / (df['Zr']  + eps)
    df['La/Zr']   = df['La']  / (df['Zr']  + eps)
    df['La/Yb']   = df['La']  / (df['Yb']  + eps)
    df['Sr/Y']    = df['Sr']  / (df['Y']   + eps)
    df['Gd/Yb']   = df['Gd']  / (df['Yb']  + eps)
    df['Nd/Nd*']  = df['Nd']  / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb']   = df['Dy']  / (df['Yb']  + eps)
    df['Th/Yb']   = df['Th']  / (df['Yb']  + eps)
    df['Nb/Yb']   = df['Nb']  / (df['Yb']  + eps)
    df['Zr/Y']    = df['Zr']  / (df['Y']   + eps)
    df['Ti/Zr']   = df['Ti']  / (df['Zr']  + eps)
    df['Y/Nb']    = df['Y']   / (df['Nb']  + eps)
    df['Th/Nb']   = df['Th']  / (df['Nb']  + eps)
    if 'Ti' in df and 'V' in df:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Ti' in df and 'Al' in df:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)
    if 'Dy/Yb' in df and 'La/Yb' in df:
        df['Crustal_Thickness_Index'] = df['Dy/Yb']
        df['Melting_Depth_Index']      = df['La/Yb']
    if 'Th/Nb' in df:
        df['Crustal_Contamination_Index'] = df['Th/Nb']
    if 'Nb/Y' in df:
        df['Mantle_Fertility_Index']      = df['Nb/Y']
    if 'Sm' in df and 'Nd' in df:
        df['Sm/Nd'] = df['Sm'] / (df['Nd'] + eps)
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("Load split GEOROC CSVs:")
use_github = st.checkbox("⬇️ Load cleaned GEOROC parts from GitHub")
uploads   = st.file_uploader(
    "📂 Or upload your own split GEOROC CSVs",
    type="csv",
    accept_multiple_files=True
)

df_list = []
skipped  = []

if use_github:
    import requests
    for i in range(1,6):
        url = f"https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part{i}.csv"
        try:
            txt  = requests.get(url).text
            part = load_data(io.StringIO(txt))       # <<< apply load_data here
            if part.isnull().any().any():
                skipped.append(f"part{i}")
            else:
                df_list.append(part)
        except:
            skipped.append(f"part{i}")

elif uploads:
    for f in uploads:
        try:
            part = load_data(f)
            if part.isnull().any().any():
                skipped.append(f.name)
            else:
                df_list.append(part)
        except:
            skipped.append(f.name)
else:
    st.warning("📥 Please either tick the GitHub box or upload at least one CSV.")
    st.stop()

if not df_list:
    st.error(f"❌ Could not load any valid GEOROC parts. Skipped: {skipped}")
    st.stop()

# concatenate + filter + ratios
df = pd.concat(df_list, ignore_index=True)
df = filter_basaltic(df)     # now 'SiO2' always exists
df = compute_ratios(df)

if skipped:
    st.warning(f"⚠ Skipped files: {', '.join(skipped)}")

# -----------------------------
# REE Spider Plot
# -----------------------------
CHONDRITE = {'Ce':1.59,'Nd':1.23,'Dy':0.26,'Yb':0.17,'Gd':0.27}
REE_ELEM  = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
if all(e in df.columns for e in REE_ELEM):
    fig, ax = plt.subplots()
    for _, row in df.iterrows():
        normed = [row[e]/CHONDRITE[e] for e in REE_ELEM]
        ax.plot(REE_ELEM, normed, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylabel('Chondrite-normalized')
    ax.set_title('REE Spider Diagrams')
    st.pyplot(fig)

# -----------------------------
# Ratios Summary
# -----------------------------
ratios = [c for c in [
    'Th/La','Ce/Ce*','Nb/Zr','La/Zr','La/Yb','Sr/Y','Gd/Yb',
    'Nd/Nd*','Dy/Yb','Th/Nb','Nb/Yb','Ti/V','Ti/Al'
] if c in df.columns]
if ratios:
    st.subheader("Computed Ratios Summary")
    st.dataframe(df[ratios].describe().T.style.format("{:.2f}"))
else:
    st.info("No ratios computed—check required elements.")

# -----------------------------
# Export
# -----------------------------
if st.button("Download Processed CSV"):
    data = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download", data, "processed_georoc.csv", "text/csv")

# -----------------------------
# Supervised Classification
# -----------------------------
st.subheader("ML Classification: Tectonic Setting")

# drop any stray isotope columns from features
exclude = {
    '143Nd/144Nd','176Hf/177Hf','87Sr/86Sr','206Pb/204Pb','207Pb/204Pb',
    'εNd','εHf','Pb_isotope_ratio'
}
num_cols = df.select_dtypes(include=np.number).columns
features = [c for c in num_cols if c not in exclude]

label_col = st.selectbox(
    "Label Column",
    [c for c in df.columns if df[c].nunique()<20]
)
train_feats = st.multiselect(
    "Features", features, default=features[:10]
)

if label_col and train_feats:
    df_cls = df.dropna(subset=[label_col]+train_feats)
    if len(df_cls) < 2:
        st.error(f"Not enough samples ({len(df_cls)}) for train/test split.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df_cls[train_feats],
            df_cls[label_col].astype('category').cat.codes,
            test_size=0.3, random_state=42
        )
        model = MLPClassifier((50,25), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba  = model.predict_proba(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        dfp = pd.DataFrame(proba, columns=pd.Categorical(df_cls[label_col]).categories)
        dfp["True"]      = pd.Categorical(df_cls[label_col]).categories[y_test]
        dfp["Predicted"] = pd.Categorical(df_cls[label_col]).categories[y_pred]

        st.subheader("Prediction Probabilities")
        st.dataframe(dfp.head())
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # fill in missing labels
        df["Auto_Label"] = None
        mask = df[label_col].isna()
        df.loc[mask, "Auto_Label"] = model.predict(df.loc[mask, train_feats])
