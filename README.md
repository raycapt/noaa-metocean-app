# Streamlit NOAA Met-Ocean (Cloud-optimized)

This is tuned for **Streamlit Community Cloud**:
- Uses **pydap** engine for OPeNDAP/ERDDAP
- Converts timestamps to **naive UTC** before dataset selection
- Wind fallback uses **NOAA PSL NCEP/NCAR Reanalysis** when older than NOMADS retention

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
