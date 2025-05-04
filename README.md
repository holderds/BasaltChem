# Basalt Geochemistry Interpreter

A Streamlit app for interpreting basalt and basaltic andesite geochemistry using:
- REE patterns and tectonic discrimination diagrams
- Isotope ratios: εNd, εHf
- PCA, t-SNE, KMeans, DBSCAN
- Supervised ML classification
- Geochemical proxy indices: La/Yb, Dy/Yb, Th/Nb, Nb/Y, etc.

## Run Locally
```
pip install -r requirements.txt
streamlit run basalt_geochem_pipeline.py
```

## Deploy on Streamlit Cloud
1. Upload this project to a public GitHub repo
2. Go to https://share.streamlit.io
3. Select the repo and `basalt_geochem_pipeline.py`
4. Click **Deploy**
