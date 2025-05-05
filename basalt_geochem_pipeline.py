# basalt_geochem_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# --- Configuration ----------------------------------------------------------

CHONDRITE_VALUES = {
    'La': 0.237, 'Ce': 0.613, 'Pr': 0.0928, 'Nd': 0.512, 'Sm': 0.153,
    'Eu': 0.056, 'Gd': 0.199, 'Tb': 0.036, 'Dy': 0.157, 'Ho': 0.035,
    'Er': 0.082, 'Tm': 0.012, 'Yb': 0.082, 'Lu': 0.012
}

REE_ELEMENTS = list(CHONDRITE_VALUES.keys())

MAJOR_COLUMNS = ['SiO2','Ti','Al','Fe2O3','MgO','CaO','Na2O','K2O','P2O5']
TRACE_COLUMNS = ['Th','Nb','Zr','Y','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','V','Sc','Co','Ni','Cr','Hf']

ALL_RAW_COLUMNS = {
    'SiO2 [wt%]':'SiO2', 'TiO2 [wt%]':'Ti', 'Al2O3 [wt%]':'Al', 'Fe2O3(t) [wt%]':'Fe2O3',
    'MgO [wt%]':'MgO','CaO [wt%]':'CaO','Na2O [wt%]':'Na2O','K2O [wt%]':'K2O',
    'P2O5 [wt%]':'P2O5','Th [ppm]':'Th','Nb [ppm]':'Nb','Zr [ppm]':'Zr','Y [ppm]':'Y',
    'La [ppm]':'La','Ce [ppm]':'Ce','Pr [ppm]':'Pr','Nd [ppm]':'Nd','Sm [ppm]':'Sm',
    'Eu [ppm]':'Eu','Gd [ppm]':'Gd','Tb [ppm]':'Tb','Dy [ppm]':'Dy','Ho [ppm]':'Ho',
    'Er [ppm]':'Er','Tm [ppm]':'Tm','Yb [ppm]':'Yb','Lu [ppm]':'Lu','V [ppm]':'V',
    'Sc [ppm]':'Sc','Co [ppm]':'Co','Ni [ppm]':'Ni','Cr [ppm]':'Cr','Hf [ppm]':'Hf'
}

# --- Helper functions -------------------------------------------------------

def load_and_clean(source):
    """Load a CSV (filelike or URL), rename columns, coerce to numeric, drop all‐NaN rows."""
    df = pd.read_csv(source, low_memory=False)
    df = df.rename(columns=ALL_RAW_COLUMNS)
    # Keep only columns we care about
    want = [c for c in ALL_RAW_COLUMNS.values() if c in df.columns]
    df = df[want].copy()
    # Coerce everything numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop rows where all majors are missing
    if 'SiO2' in df:
        df = df.dropna(subset=['SiO2'], how='all')
    return df

def filter_basaltic(df):
    """Keep only samples with 45–57 wt% SiO2."""
    if 'SiO2' not in df.columns:
        st.error("SiO2 column missing—check your data.")
        st.stop()
    return df.loc[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    """Add major/trace and REE ratios."""
    eps = 1e-6
    # Basic ratios
    for a,b in [('Th','La'),('Nb','Zr'),('La','Zr'),('La','Yb'),('Sr','Y'),
                ('Gd','Yb'),('Dy','Yb'),('Th','Nb'),('Nb','Yb')]:
        if a in df and b in df:
            df[f"{a}/{b}"] = df[a] / (df[b] + eps)
    # Ce/Ce*, Eu/Eu* etc.
    if {'La','Ce','Pr'}.issubset(df):
        df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    if {'Sm','Nd','Pr'}.issubset(df):
        df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    if {'Eu','Sm','Gd'}.issubset(df):
        df['Eu/Eu*'] = df['Eu'] / (np.sqrt(df['Sm'] * df['Gd']) + eps)
    # REE spider normalization
    for re in REE_ELEMENTS:
        if re in df:
            df[f"{re}_norm"] = df[re] / CHONDRITE_VALUES[re]
    return df

# --- Streamlit UI -----------------------------------------------------------

st.title("Basalt & Basaltic Andesite Geochemistry Interpreter")

st.write("Load GEOROC parts from GitHub:")
use_zip = st.checkbox("▶ Load example GEOROC parts from GitHub")
uploaded = st.file_uploader("Or upload your own CSV files", type="csv", accept_multiple_files=True)
use_example = st.checkbox("▶ Use single cleaned example CSV")

df = None
skipped = []

if use_zip:
    # URLs of your five parts
    urls = [f"https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_part{i}.csv"
            for i in range(1,6)]
    parts = []
    for url in urls:
        try:
            parts.append(load_and_clean(url))
        except Exception:
            skipped.append(url.split("/")[-1])
    if not parts:
        st.error("❌ Could not load any remote parts.")
        st.stop()
    df = pd.concat(parts, ignore_index=True)

elif uploaded:
    parts = []
    for up in uploaded:
        try:
            parts.append(load_and_clean(up))
        except Exception:
            skipped.append(up.name)
    if not parts:
        st.error("❌ Could not load any uploaded files.")
        st.stop()
    df = pd.concat(parts, ignore_index=True)

elif use_example:
    # single example bundled CSV in repo
    df = load_and_clean("https://raw.githubusercontent.com/holderds/basaltchem/main/example_data/cleaned_georoc_basalt_example.csv")

else:
    st.warning("Please choose a data source above.")
    st.stop()

if skipped:
    st.warning("Skipped files: " + ", ".join(skipped))

# Filter & compute
df = filter_basaltic(df)
df = compute_ratios(df)

# --- Plotting ---------------------------------------------------------------

st.subheader("εNd vs εHf (if available)")
if {'εNd','εHf'}.issubset(df):
    fig,ax = plt.subplots()
    ax.scatter(df['εNd'], df['εHf'], alpha=0.6)
    ax.set_xlabel("εNd"); ax.set_ylabel("εHf")
    st.pyplot(fig)

st.subheader("REE Spider Plot")
if all(f"{re}_norm" in df for re in REE_ELEMENTS):
    fig,ax = plt.subplots()
    for _,row in df.iterrows():
        ax.plot(REE_ELEMENTS, [row[f"{re}_norm"] for re in REE_ELEMENTS], alpha=0.3)
    ax.set_yscale('log'); ax.set_ylabel("Normalized to Chondrite")
    st.pyplot(fig)

st.subheader("Computed Ratios Summary")
ratios = [c for c in df.columns if "/" in c]
if ratios:
    st.dataframe(df[ratios].describe().T.style.format("{:.2f}"))
else:
    st.write("No ratios computed—check that your data contain the required elements.")

# --- Export -----------------------------------------------------------------

if st.button("Download processed data as CSV"):
    b = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", data=b, file_name="processed_basalt.csv")

# --- Machine Learning Classification ---------------------------------------

st.subheader("ML Classification: Tectonic Setting")

# choose any categorical column with <20 classes
labels = [c for c in df.columns if df[c].nunique(dropna=True) < 20]
label = st.selectbox("Label column", labels)

features = st.multiselect("Features (numeric)", df.select_dtypes("number").columns.tolist())
if label and features:
    df_class = df.dropna(subset=features + [label])
    X = df_class[features]
    y = df_class[label].astype("category")

    if len(df_class) < 2:
        st.error("Not enough samples for ML split.")
    else:
        X_train,X_test,y_train,y_test = train_test_split(X, y.cat.codes, test_size=0.3, random_state=42)

        clf = MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # probabilities with true/pred labels
        proba = (
            pd.DataFrame(clf.predict_proba(X_test), columns=y.cat.categories)
              .assign(
                  True_Label=y.cat.categories[y_test.values],
                  Predicted_Label=y.cat.categories[y_pred]
              )
        )

        st.subheader("Prediction Probabilities")
        st.dataframe(proba.head())
        st.write("### Classification Report")
        st.text(clf.classes_)
        st.text(pd.DataFrame(clf.predict(X_test), columns=["Predicted"]).describe())

