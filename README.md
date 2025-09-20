# Nautical Weather Map — Streamlit + Stormglass

This Streamlit app retrieves metocean data (wind, **Significant wave (Hs)**, **wind wave**, swell, currents) from **Stormglass** for one or many positions/timestamps and visualizes them on a **nautical map** (OpenSeaMap overlay). Bulk CSV/XLSX uploads and **CSV download** supported.

## Features
- Enter a single **timestamp + lat + lon**, or **upload CSV/XLSX** with multiple rows.
- Fetch weather from **Stormglass** for each point/time.
- Show points on a **nautical map**, **color-coded by wind speed (knots)**:
  - `< 16 kt` = green
  - `16–24 kt` = orange
  - `> 24 kt` = red
- **Hover** to see wind, **Significant wave (Hs)**, **Wind wave**, swell, and current details.
- **Download** the full enriched dataset as CSV.
- Robust timestamp parsing (accepts many formats; auto-corrects common issues).
- Caching to avoid repeat API calls during a session.
- Optional choice of Stormglass **source** (default 'best').

> ⚠️ Stormglass free tier has request limits. For large uploads, consider filtering or running in batches.

## Quick Start (Local)
1. **Python 3.10+** recommended.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your Stormglass API key to **Streamlit secrets**:
   - Create `.streamlit/secrets.toml` (or set env var `STORMGLASS_API_KEY`):
     ```toml
     STORMGLASS_API_KEY = "YOUR_KEY_HERE"
     ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud
- Push this folder to GitHub, then create a new Streamlit Cloud app pointing to `app.py`.
- On Streamlit Cloud, set the secret **STORMGLASS_API_KEY** under *App → Settings → Secrets*.

## File formats
### Single entry (UI)
- Timestamp (UTC), Latitude, Longitude.

### Bulk CSV/XLSX
Provide columns (case-insensitive; extra columns are kept and passed through):
- `timestamp` — any parseable datetime (assumed UTC if no tz)
- `lat` — latitude
- `lon` — longitude

See **sample_data.csv** for an example.

## Variables & Units
- **Significant wave height (Hs)** from `waveHeight` → `sigWaveHeight_m` (**meters**).
- **Sig wave direction** from `waveDirection` → `sigWaveDir_deg_from` (**degrees, coming from**).
- **Wind wave height** from `windWaveHeight` → `windWaveHeight_m` (**meters**) *(if available)*.
- **Wind wave direction** from `windWaveDirection` → `windWaveDir_deg_from` (**degrees, coming from**) *(if available)*.
- **Swell height/direction** from `swellHeight`/`swellDirection` (meters, degrees coming from).
- **Wind speed/direction** from `windSpeed`/`windDirection` (speed in **knots**, direction **coming from**).
- **Current speed/direction** from `currentSpeed`/`currentDirection` (speed in **knots**, direction **going to**).

Stormglass returns SI (m/s, m). The app converts speeds to knots.

## Troubleshooting
- If values are missing, try a different `Source` in the sidebar (`best`, `noaa`, `icon`, `meteo`) or use the nearest hour to your timestamp.
- The app queries the **nearest hour** window for each timestamp in UTC.

---

Built with ❤️ for practical voyage & performance analysis.
