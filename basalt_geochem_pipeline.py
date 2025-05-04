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
# Configuration
# -----------------------------
CHONDRITE_VALUES = {
    'Ce': 1.59, 'Nd': 1.23, 'Dy': 0.26, 'Yb': 0.17, 'Gd': 0.27
}

REE_ELEMENTS = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
IMMOBILE_ELEMENTS = ['Ti', 'Al', 'Zr', 'Nb', 'Y', 'Th', 'Sc', 'Co', 'Ni', 'Cr', 'Hf']

# -----------------------------
# Load and Filter Data (removed duplicate definitions)
# -----------------------------

# -----------------------------
# Removed duplicate compute_ratios block
# -----------------------------

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
    df.loc[df[label_column].isna(), 'Auto_Label'] = model.predict(df.loc[df[label_column].isna(), train_features])
    df['Predicted_Class'] = pd.Series(model.predict(X), index=df_class.index)
    df['Label'] = y

# -----------------------------
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

@st.cache_data
def run_pca_tsne(df, features):
    df_clean = df.dropna(subset=features)
    X = df_clean[features].values

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df_clean['PCA1'] = pca_result[:, 0]
    df_clean['PCA2'] = pca_result[:, 1]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X)
    df_clean['TSNE1'] = tsne_result[:, 0]
    df_clean['TSNE2'] = tsne_result[:, 1]

    return df_clean

# -----------------------------
# εNd vs εHf Plot
# -----------------------------
st.subheader("εNd vs εHf (Isotope Discrimination)")
if 'εNd' in df.columns and 'εHf' in df.columns:
    fig_iso, ax_iso = plt.subplots()
    for i, row in df.dropna(subset=['εNd', 'εHf']).iterrows():
        label = str(row['Sample ID']) if 'Sample ID' in df.columns else str(i)
        ax_iso.text(row['εNd'], row['εHf'], label, fontsize=7, alpha=0.6)
    ax_iso.scatter(df['εNd'], df['εHf'], alpha=0.7, edgecolors='k')

    # Plot approximate global reference fields
    ax_iso.axhline(y=15, color='gray', linestyle='--', linewidth=0.8)
    ax_iso.axhline(y=10, color='gray', linestyle='--', linewidth=0.8)
    ax_iso.axhline(y=5, color='gray', linestyle='--', linewidth=0.8)
    ax_iso.axvline(x=5, color='gray', linestyle='--', linewidth=0.8)
    ax_iso.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

    ax_iso.fill_between([-5, 8], [8, 16], [12, 20], color='lightblue', alpha=0.3, label='Depleted Mantle Field')
    ax_iso.fill_between([-5, 0], [4, 6], [12, 14], color='peachpuff', alpha=0.3, label='Enriched Mantle Field')
    ax_iso.fill_between([-10, 2], [0, 2], [6, 8], color='lightgray', alpha=0.3, label='Continental Crust')

    ax_iso.set_xlabel("εNd")
    ax_iso.set_ylabel("εHf")
    ax_iso.set_title("εNd vs εHf")
    ax_iso.legend()
    st.pyplot(fig_iso)
    buf_iso = io.BytesIO()
    fig_iso.savefig(buf_iso, format='png')
    buf_iso.seek(0)
    st.download_button("Download εNd vs εHf Plot", data=buf_iso, file_name="epsilon_nd_hf_plot.png", mime="image/png")

# -----------------------------
# Streamlit PCA/t-SNE
# -----------------------------
st.subheader("PCA and t-SNE Projection")
all_features = [col for col in df.columns if col not in ['Label'] and df[col].dtype in [np.float64, np.float32, np.int64]]
selected_feats = st.multiselect("Select features for PCA/t-SNE", all_features, default=all_features[:10])

if selected_feats:
    proj_df = run_pca_tsne(df, selected_feats)

    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(proj_df['PCA1'], proj_df['PCA2'], alpha=0.6)
    ax_pca.set_xlabel("PCA1")
    ax_pca.set_ylabel("PCA2")
    ax_pca.set_title("PCA Projection")
    st.pyplot(fig_pca)

    fig_tsne, ax_tsne = plt.subplots()
    ax_tsne.scatter(proj_df['TSNE1'], proj_df['TSNE2'], alpha=0.6)
    ax_tsne.set_xlabel("t-SNE1")
    ax_tsne.set_ylabel("t-SNE2")
    ax_tsne.set_title("t-SNE Projection")
    st.pyplot(fig_tsne)

# -----------------------------
# Clustering (KMeans) with Class Coloring
# -----------------------------
from sklearn.cluster import KMeans

st.subheader("Clustering (KMeans)")
color_by = st.selectbox("Color points by", options=['Label', 'Cluster', 'DBSCAN'], index=0)
num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

if selected_feats:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    proj_df['Cluster'] = kmeans.fit_predict(proj_df[selected_feats])

    fig_kmeans_pca, ax_kmeans_pca = plt.subplots()
    scatter = ax_kmeans_pca.scatter(proj_df['PCA1'], proj_df['PCA2'], proj_df[color_by].astype('category').cat.codes if color_by in proj_df.columns else proj_df['Cluster'], cmap='tab10', alpha=0.7)
    ax_kmeans_pca.set_xlabel("PCA1")
    ax_kmeans_pca.set_ylabel("PCA2")
    ax_kmeans_pca.set_title("KMeans Clustering on PCA")
    handles, labels = scatter.legend_elements(prop='colors', alpha=0.6)
    ax_kmeans_pca.legend(handles, proj_df[color_by].astype('category').cat.categories if color_by in proj_df.columns else [f'Cluster {i}' for i in range(num_clusters)], title="Class")
    st.pyplot(fig_kmeans_pca)
    buf_pca = io.BytesIO()
    fig_kmeans_pca.savefig(buf_pca, format='png')
    buf_pca.seek(0)
    st.download_button("Download PCA Plot", data=buf_pca, file_name="pca_plot.png", mime="image/png")

    fig_kmeans_tsne, ax_kmeans_tsne = plt.subplots()
    scatter = ax_kmeans_tsne.scatter(proj_df['TSNE1'], proj_df['TSNE2'], c=proj_df['Label'].astype('category').cat.codes if 'Label' in proj_df.columns else proj_df['Cluster'], cmap='tab10', alpha=0.7)
    ax_kmeans_tsne.set_xlabel("t-SNE1")
    ax_kmeans_tsne.set_ylabel("t-SNE2")
    ax_kmeans_tsne.set_title("KMeans Clustering on t-SNE")
    handles, labels = scatter.legend_elements(prop='colors', alpha=0.6)
    ax_kmeans_tsne.legend(handles, proj_df['Label'].astype('category').cat.categories if 'Label' in proj_df.columns else [f'Cluster {i}' for i in range(num_clusters)], title="Class")
    st.pyplot(fig_kmeans_tsne)
    buf_tsne = io.BytesIO()
    fig_kmeans_tsne.savefig(buf_tsne, format='png')
    buf_tsne.seek(0)
    st.download_button("Download t-SNE Plot", data=buf_tsne, file_name="tsne_plot.png", mime="image/png")

# -----------------------------
# REE Spider Plot by Tectonic Group
# -----------------------------
st.subheader("REE Spider Plot by Tectonic Group")
group_col = st.selectbox("Group by column", options=[col for col in df.columns if df[col].nunique() < 20], index=0)
ree_cols = [el for el in REE_ELEMENTS if el in df.columns]
if ree_cols and group_col:
    fig_ree, ax_ree = plt.subplots()
    for name, group in df.groupby(group_col):
        mean_vals = group[ree_cols].mean()
        ax_ree.plot(ree_cols, mean_vals, label=str(name), marker='o')
    ax_ree.set_yscale('log')
    ax_ree.set_title(f"Mean REE Pattern by {group_col}")
    ax_ree.set_ylabel("REE (ppm)")
    ax_ree.legend()
    st.pyplot(fig_ree)
    buf_ree = io.BytesIO()
    fig_ree.savefig(buf_ree, format='png')
    buf_ree.seek(0)
    st.download_button("Download REE Spider Plot", data=buf_ree, file_name="ree_spider_plot.png", mime="image/png")

# -----------------------------
# Export Clustered Data
# -----------------------------
st.subheader("Download Clustered Data")
if 'Cluster' in proj_df.columns:
    csv = proj_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv,
        file_name='clustered_geochem_data.csv',
        mime='text/csv'
    )

# -----------------------------
# Optional: DBSCAN Clustering
# -----------------------------
from sklearn.cluster import DBSCAN
st.subheader("Optional Clustering (DBSCAN)")
eps_val = st.slider("DBSCAN eps", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
min_samples_val = st.slider("Min Samples", min_value=1, max_value=20, value=5)

if selected_feats:
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    proj_df['DBSCAN'] = dbscan.fit_predict(proj_df[selected_feats])

    fig_db_pca, ax_db_pca = plt.subplots()
    scatter = ax_db_pca.scatter(proj_df['PCA1'], proj_df['PCA2'], c=proj_df['DBSCAN'], cmap='Spectral', alpha=0.7)
    ax_db_pca.set_xlabel("PCA1")
    ax_db_pca.set_ylabel("PCA2")
    ax_db_pca.set_title("DBSCAN Clustering on PCA")
    st.pyplot(fig_db_pca)
    buf_db_pca = io.BytesIO()
    fig_db_pca.savefig(buf_db_pca, format='png')
    buf_db_pca.seek(0)
    st.download_button("Download DBSCAN PCA Plot", data=buf_db_pca, file_name="dbscan_pca_plot.png", mime="image/png")

    fig_db_tsne, ax_db_tsne = plt.subplots()
    scatter = ax_db_tsne.scatter(proj_df['TSNE1'], proj_df['TSNE2'], c=proj_df['DBSCAN'], cmap='Spectral', alpha=0.7)
    ax_db_tsne.set_xlabel("t-SNE1")
    ax_db_tsne.set_ylabel("t-SNE2")
    ax_db_tsne.set_title("DBSCAN Clustering on t-SNE")
    st.pyplot(fig_db_tsne)
    buf_db_tsne = io.BytesIO()
    fig_db_tsne.savefig(buf_db_tsne, format='png')
    buf_db_tsne.seek(0)
    st.download_button("Download DBSCAN t-SNE Plot", data=buf_db_tsne, file_name="dbscan_tsne_plot.png", mime="image/png")

# -----------------------------
# Requirements for Standalone Deployment
# -----------------------------
# Create a file named requirements.txt with the following contents:
#
# pandas
# numpy
# matplotlib
# scikit-learn
# streamlit

# Save the main script as basalt_geochem_app.py and run locally with:
# streamlit run basalt_geochem_app.py

# To share, deploy to Streamlit Cloud or similar platforms.
# [rest of code remains unchanged]
