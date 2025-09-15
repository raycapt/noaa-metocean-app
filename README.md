# Streamlit NOAA Met-Ocean (Cloud v2)

Adds:
- Nearest-ocean sampling for waves & currents (avoids land-mask NaNs near coast)
- Robust variable names for PacIOOS WW3 (wind-wave fields vary)
- If wind-wave height missing, estimate from Hs and swell via quadratic energy difference
