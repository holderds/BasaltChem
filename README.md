# Basalt Geochemistry Interpreter

This app allows geochemical interpretation of basaltic rocks:
- REE patterns, εNd vs εHf isotopes
- Crustal thickness, contamination, melting depth proxies
- PCA, t-SNE, clustering
- MLP classification of tectonic settings

## Run Locally
```bash
pip install -r requirements.txt
streamlit run basalt_geochem_pipeline.py
```

## Deploy on Streamlit Cloud
1. Upload to GitHub
2. Go to https://share.streamlit.io
3. Select `basalt_geochem_pipeline.py`
4. Deploy
