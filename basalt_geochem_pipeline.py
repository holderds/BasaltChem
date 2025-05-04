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
# Move Function Definitions to Top
# -----------------------------

def load_data(upload):
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

def filter_basalt_to_basaltic_andesite(df):
    return df[(df['SiO2'] >= 45) & (df['SiO2'] <= 57)]

def compute_ratios(df):
    eps = 1e-6
    if 'La/Yb' in df.columns and 'Dy/Yb' in df.columns:
        df['Crustal_Thickness_Index'] = df['Dy/Yb']
        df['Melting_Depth_Index'] = df['La/Yb']
    if 'Th/Nb' in df.columns and 'Th/Yb' in df.columns:
        df['Crustal_Contamination_Index'] = df['Th/Nb']
    if 'Nb/Y' in df.columns and 'La/Yb' in df.columns:
        df['Mantle_Fertility_Index'] = df['Nb/Y']
    if 'Sm' in df.columns and 'Nd' in df.columns:
        df['Sm/Nd'] = df['Sm'] / df['Nd']
    if '143Nd/144Nd' in df.columns:
        CHUR_Nd = 0.512638
        df['εNd'] = ((df['143Nd/144Nd'] - CHUR_Nd) / CHUR_Nd) * 1e4
    if '176Hf/177Hf' in df.columns:
        CHUR_Hf = 0.282785
        df['εHf'] = ((df['176Hf/177Hf'] - CHUR_Hf) / CHUR_Hf) * 1e4
    if '87Sr/86Sr' in df.columns and '143Nd/144Nd' in df.columns:
        pass
    if '206Pb/204Pb' in df.columns and '207Pb/204Pb' in df.columns:
        df['Pb_isotope_ratio'] = df['206Pb/204Pb'] / df['207Pb/204Pb']
    df['Th/La'] = df['Th'] / (df['La'] + eps)
    df['Ce/Ce*'] = df['Ce'] / (np.sqrt(df['La'] * df['Pr']) + eps)
    df['Nb/Zr'] = df['Nb'] / (df['Zr'] + eps)
    df['La/Zr'] = df['La'] / (df['Zr'] + eps)
    df['La/Yb'] = df['La'] / (df['Yb'] + eps)
    df['Sr/Y'] = df['Sr'] / (df['Y'] + eps)
    df['Gd/Yb'] = df['Gd'] / (df['Yb'] + eps)
    df['Nd/Nd*'] = df['Nd'] / (np.sqrt(df['Pr'] * df['Sm']) + eps)
    df['Dy/Yb'] = df['Dy'] / (df['Yb'] + eps)
    df['Th/Yb'] = df['Th'] / (df['Yb'] + eps)
    df['Nb/Yb'] = df['Nb'] / (df['Yb'] + eps)
    df['Zr/Y'] = df['Zr'] / (df['Y'] + eps)
    df['Ti/Zr'] = df['Ti'] / (df['Zr'] + eps)
    df['Y/Nb'] = df['Y'] / (df['Nb'] + eps)
    df['Th/Nb'] = df['Th'] / (df['Nb'] + eps)
    if 'Ti' in df.columns and 'V' in df.columns:
        df['Ti/V'] = df['Ti'] / (df['V'] + eps)
    if 'Ti' in df.columns and 'Al' in df.columns:
        df['Ti/Al'] = df['Ti'] / (df['Al'] + eps)
    return df

# -----------------------------
# Streamlit Upload and Initialization
# -----------------------------
st.title("Basalt and Basaltic Andesite Geochemistry Interpreter")

uploaded_file = st.file_uploader("Upload your geochemical CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        df = filter_basalt_to_basaltic_andesite(df)
        df = compute_ratios(df)
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# -----------------------------
# εNd vs εHf Isotope Plot
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['εNd'], df['εHf'], c='blue', alpha=0.6)
    ax.set_xlabel("εNd")
    ax.set_ylabel("εHf")
    ax.set_title("εNd vs εHf Isotope Plot")
    st.pyplot(fig)
    st.session_state['last_figure'] = fig

# -----------------------------
# Configuration
# -----------------------------
CHONDRITE_VALUES = {
    'Ce': 1.59, 'Nd': 1.23, 'Dy': 0.26, 'Yb': 0.17, 'Gd': 0.27
}

REE_ELEMENTS = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
IMMOBILE_ELEMENTS = ['Ti', 'Al', 'Zr', 'Nb', 'Y', 'Th', 'Sc', 'Co', 'Ni', 'Cr', 'Hf']

# -----------------------------
# Export Results
# -----------------------------
# Download labeled data CSV
if st.button("Download Predictions as CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Labeled Data",
        data=csv,
        file_name='classified_geochem_data.csv',
        mime='text/csv'
    )

# Export last figure as PNG
if 'last_figure' in st.session_state:
    buf = io.BytesIO()
    st.session_state['last_figure'].savefig(buf, format="png")
    st.download_button(
        label="📷 Download Last Plot as PNG",
        data=buf.getvalue(),
        file_name="last_plot.png",
        mime="image/png"
    )

# -----------------------------
# Machine Learning Classification
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

st.subheader("Supervised Classification (Tectonic Setting)")
label_column = st.selectbox("Select Label Column", options=[col for col in df.columns if df[col].nunique() < 20], index=0)
all_features = [col for col in df.columns if df[col].dtype in [np.float64, np.float32, np.int64]]
selected_feats = all_features[:10]
train_features = st.multiselect("Select features for classification", options=all_features, default=selected_feats)

if train_features and label_column:
    df_class = df.dropna(subset=train_features + [label_column])
    X = df_class[train_features]
    y = df_class[label_column].astype('category')
    y_cat = y.cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
    y_proba = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_proba = pd.DataFrame(y_proba, columns=y.cat.categories)
    df_proba['True_Label'] = y.cat.categories[y_test].values
    df_proba['Predicted_Label'] = y.cat.categories[y_pred].values
    st.subheader("Prediction Probabilities")
    st.dataframe(df_proba.head())
    st.text("MLPClassifier Performance:")
    st.dataframe(pd.DataFrame(report).transpose())

    # Assign predicted label codes back to full dataset
    df['Auto_Label'] = None
    missing_label_rows = df[label_column].isna()
    if missing_label_rows.any():
        df.loc[missing_label_rows, 'Auto_Label'] = model.predict(df.loc[missing_label_rows, train_features])
    df['Predicted_Class'] = pd.Series(model.predict(X), index=df_class.index)
    df['Label'] = y

# -----------------------------
# [rest of code remains unchanged]
