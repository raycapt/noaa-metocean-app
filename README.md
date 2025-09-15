# Streamlit NOAA Met-Ocean — Final Pack

Features:
- **Two modes**: Single point *or* CSV/XLSX upload
- **Wind (10 m)**: NOMADS WW3 recent; fallback **NOAA PSL reanalysis** (global)
- **Waves (WW3)**: NOMADS WW3 (recent) or **PacIOOS WW3 Global** (continuous)
- **Currents**: **CoastWatch Blended NRT** via ERDDAP `.dods`; fallback to **OSCAR** if masked
- **Nearest-ocean sampling** with expanding search window (up to ±5°) and sampled cell shown on map
- Map dots colored by wind: <16 kt green, 16–25 kt orange, >25 kt red

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
- Push these files to GitHub
- On https://share.streamlit.io → **New app** → select repo → file `app.py`
- Python 3.10/3.11 recommended

## CSV template

```csv
timestamp,lat,lon
2025-07-10 06:00,24.5,54.6
2024-11-03 12:00,12.0,130.0
```

## Notes
- Directions: Wind **FROM** (°T), currents **TO** (°T), wave/swell **FROM** (°T).
- If a point lies on land or a masked pixel, the app samples the nearest valid ocean grid cell and displays the sampled lat/lon.
