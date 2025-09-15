# Streamlit NOAA Met-Ocean (Cloud v3)

This version hardens **currents**:
- Auto-detects ERDDAP variable names (u/v current) across datasets
- Handles masked arrays from pydap correctly
- Nearest valid ocean cell search expands from ±0.75° up to ±2° if needed

Also keeps:
- Nearest-ocean sampling for waves
- Robust PacIOOS WW3 variable discovery
