# Met‑Ocean Streamlit (v8) — ERDDAP JSON only

Fetch **wind (kts, °T from)**, **significant wave height**, **wind‑sea & swell** (m and °T from), and **surface current** (kts, °T going-to) at points worldwide (since 2024→today).

## Data sources (ERDDAP JSON)
- **Waves** (WW3 Global, PacIOOS): `Thgt, Tdir, whgt, wdir, shgt, sdir`
- **Wind @10 m** (CoastWatch Blended Winds Daily): `windspeed, u_wind, v_wind` (daily @ 00Z)
- **Surface currents** (CoastWatch Blended NRT) → fallback **OSCAR**: `u_current/v_current` or `u/v` (daily @ 00Z)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
- **Python**: 3.11
- App file: `app.py`

## Input
- Single: timestamp `YYYY-MM-DD HH:MM` (UTC), lat, lon.
- Batch: CSV/XLSX with `timestamp,lat,lon`.

**Directions**: wave/swell/wind are **FROM** (° true). currents are **TO** (° true).
