# Streamlit NOAA Met‑Ocean Point Extractor (2024 → today)

A Streamlit app that takes either **a single timestamp+position** or **a CSV/XLSX of many points** and returns, for each row:

- **Wind**: direction (°T, coming **from**), speed (kt)
- **Waves (WW3)**: wind‑wave height (m) & direction (°T, **from**), swell height (m) & direction (°T, **from**), **significant wave height** (m) & its representative **direction** (primary wave dir, °T, **from**)
- **Currents**: surface speed (kt) & direction **TO** (°T)

It plots all points on a nautical‑style map (OpenSeaMap seamarks). Dot colors by wind speed: **<16 kt green**, **16–25 kt orange**, **>25 kt red**.

## Data sources

- **Wind (10 m)**: GFS Grid‑4 Analysis via **NCEI THREDDS** (per‑hour NetCDF/GRIB over OPeNDAP). Fallback (for recent dates): **NOMADS GFS‑Wave** surface wind fields.
- **Waves**: If date is within NOMADS retention (~6 days), uses **NOMADS WW3/GFS‑Wave global 0.25°**; otherwise uses **PacIOOS WW3 Global** (NOAA‑funded IOOS) “best” aggregation.
- **Currents**: **NOAA CoastWatch ERDDAP Blended NRT Currents** (global, daily, 0.25°).

> Notes: “Significant wave height direction” isn’t a distinct variable in many products; we use **primary wave direction** as the representative direction for Hs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV template

```csv
timestamp,lat,lon
2025-07-10 06:00,24.5,54.4
2024-11-03 12:00,12.0,130.0
```

## Deploy on Replit (single command)

Set `run` in `.replit` to:
```ini
run = "bash -lc \"pip install -r requirements.txt && streamlit run app.py --server.port 8000 --server.address 0.0.0.0\""
```

## GitHub

```bash
git init
git add app.py requirements.txt README.md samples/*
git commit -m "NOAA met-ocean point extractor (Streamlit)"
git branch -M main
git remote add origin https://github.com/<you>/streamlit-noaa-metocean-app.git
git push -u origin main
```

## Disclaimer

This tool is for **informational** / **visualization** purposes only and not for safe navigation. Data services can have outages; results are retrieved using nearest grid/time and may differ from official products.
