import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Met‑Ocean at Points · 2024→today", layout="wide")
st.title("Historical NOAA Wind · Wave · Current at Positions")
st.caption("Upload a CSV/XLSX OR enter a single timestamp/position. I’ll fetch wind (10 m), waves (WW3), and surface currents.")

# =========================
# Constants / helpers
# =========================
KTS_PER_MS = 1.9438444924574

@st.cache_data(show_spinner=False)
def open_ds(url: str):
    return xr.open_dataset(url)

@lru_cache(maxsize=1024)
def gfswave_url_for(ts: datetime):
    # NOMADS GFS-Wave 0.25° for dates in the last ~6 days
    cyc = f"{(ts.hour // 6) * 6:02d}"
    return f"https://nomads.ncep.noaa.gov:9090/dods/wave/gfswave/{ts:%Y%m%d}/gfswave.global.0p25_{cyc}z"

@lru_cache(maxsize=1024)
def gfs_anl_url_for(ts: datetime):
    # NCEI THREDDS – GFS Grid-4 analysis file for the exact hour
    return f"https://www.ncei.noaa.gov/thredds/dodsC/model-gfs-g4-anl-files/{ts:%Y%m}/gfsanl_4_{ts:%Y%m%d}_{ts:%H}00_000.grb2"

@lru_cache(maxsize=1)
def pacioos_ww3_url():
    # NOAA-funded IOOS PacIOOS WW3 Global "best"
    return "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ww3_global/WaveWatch_III_Global_Wave_Model_best.ncd"

@lru_cache(maxsize=1)
def coastwatch_currents_url():
    # ERDDAP blended geostrophic currents (global, daily, 0.25°), units m/s
    return "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily.nc"

def to_0_360(lon):
    return lon if lon >= 0 else lon + 360

def nearest(ds, tname, t, yname, lat, xname, lon):
    return ds.sel({tname: t, yname: lat, xname: lon}, method="nearest")

def wind_from_uv(u, v):
    speed_ms = np.hypot(u, v)
    # Meteorological wind direction = where wind is COMING FROM (deg true)
    deg_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return float(speed_ms), float(deg_from)

def current_dir_to(u, v):
    # Direction current flows TO (deg true)
    return float((90.0 - np.degrees(np.arctan2(v, u))) % 360.0)

# =========================
# Data fetchers
# =========================
def get_waves(ts: datetime, lat: float, lon: float):
    """Return dict with:
       - wave_hgt_m, wave_dir_degT (wind-wave)
       - swell_hgt_m, swell_dir_degT
       - sig_wave_hs_m, sig_wave_dir_degT (dir ~ primary wave direction)
       Uses NOMADS WW3 when recent, else PacIOOS WW3 global.
    """
    today = datetime.now(timezone.utc).date()
    recent_cut = today - timedelta(days=6)
    try:
        if ts.date() >= recent_cut:
            url = gfswave_url_for(ts)
            ds = open_ds(url)
            lon0360 = to_0_360(lon)
            # NOMADS WW3 GRIB variable names
            hs = nearest(ds["htsgwsfc"], "time", ts, "lat", lat, "lon", lon0360)   # significant wave height (m)
            dir_sig = nearest(ds["dirpwsfc"], "time", ts, "lat", lat, "lon", lon0360)  # primary wave dir (deg true, FROM)
            wvh = nearest(ds["wvhgtsfc"], "time", ts, "lat", lat, "lon", lon0360)  # wind wave height (m)
            wvdir = nearest(ds["wvdirsfc"], "time", ts, "lat", lat, "lon", lon0360)    # wind wave dir (deg true, FROM)
            swh = nearest(ds["swell_1"], "time", ts, "lat", lat, "lon", lon0360)       # primary swell height (m)
            swdir = nearest(ds["swdir_1"], "time", ts, "lat", lat, "lon", lon0360)     # primary swell dir (deg true, FROM)
            return {
                "source_wave": url,
                "wave_hgt_m": float(wvh.values),
                "wave_dir_degT": float(wvdir.values),
                "swell_hgt_m": float(swh.values),
                "swell_dir_degT": float(swdir.values),
                "sig_wave_hs_m": float(hs.values),
                "sig_wave_dir_degT": float(dir_sig.values),
            }
        else:
            url = pacioos_ww3_url()
            ds = open_ds(url)
            # PacIOOS WW3 uses lon in [0,360] typically
            lonx = to_0_360(lon) if ds["lon"].max() > 180 else lon
            hsig = nearest(ds["Thgt"], "time", ts, "lat", lat, "lon", lonx)  # significant wave height
            dir_sig = nearest(ds["Tdir"], "time", ts, "lat", lat, "lon", lonx)  # (deg true, FROM)
            wvh = nearest(ds["whgt"], "time", ts, "lat", lat, "lon", lonx)   # wind-wave height
            wvdir = nearest(ds["wdir"], "time", ts, "lat", lat, "lon", lonx) # wind-wave direction (FROM)
            swh = nearest(ds["shgt"], "time", ts, "lat", lat, "lon", lonx)   # swell height
            swdir = nearest(ds["sdir"], "time", ts, "lat", lat, "lon", lonx) # swell dir (FROM)
            return {
                "source_wave": url,
                "wave_hgt_m": float(wvh.values),
                "wave_dir_degT": float(wvdir.values),
                "swell_hgt_m": float(swh.values),
                "swell_dir_degT": float(swdir.values),
                "sig_wave_hs_m": float(hsig.values),
                "sig_wave_dir_degT": float(dir_sig.values),
            }
    except Exception as e:
        return {"error_wave": str(e)}

def get_wind(ts: datetime, lat: float, lon: float):
    """Prefer 10 m wind from GFS analysis (NCEI). Fallback: NOMADS gfswave surface wind when recent."""
    try:
        url = gfs_anl_url_for(ts)
        ds = open_ds(url)
        # Try typical GRIB variable names at 10 m
        cand_u = [k for k in ds.data_vars if k.lower().startswith("ugrd10m") or k.lower() == "ugrd10m"]
        cand_v = [k for k in ds.data_vars if k.lower().startswith("vgrd10m") or k.lower() == "vgrd10m"]
        if not cand_u or not cand_v:
            # fallback to any ugrd/vgrd variable (then still near 10 m in analysis)
            cand_u = [k for k in ds.data_vars if k.lower().startswith("ugrd")]
            cand_v = [k for k in ds.data_vars if k.lower().startswith("vgrd")]
        lon0360 = to_0_360(lon)
        u = nearest(ds[cand_u[0]], "time", ts, "lat", lat, "lon", lon0360).values
        v = nearest(ds[cand_v[0]], "time", ts, "lat", lat, "lon", lon0360).values
        spd_ms, dir_from = wind_from_uv(float(u), float(v))
        return {"source_wind": url, "wind_speed_kts": spd_ms * KTS_PER_MS, "wind_dir_from_degT": dir_from}
    except Exception:
        # fallback to NOMADS gfswave (recent dates only)
        try:
            url = gfswave_url_for(ts)
            ds = open_ds(url)
            lon0360 = to_0_360(lon)
            if "windsfc" in ds and "wdirsfc" in ds:
                wspd_ms = nearest(ds["windsfc"], "time", ts, "lat", lat, "lon", lon0360).values
                wdir = nearest(ds["wdirsfc"], "time", ts, "lat", lat, "lon", lon0360).values
                return {"source_wind": url, "wind_speed_kts": float(wspd_ms) * KTS_PER_MS, "wind_dir_from_degT": float(wdir)}
        except Exception as e2:
            return {"error_wind": f"No wind source reachable: {e2}"}

def get_currents(ts: datetime, lat: float, lon: float):
    try:
        url = coastwatch_currents_url()
        ds = open_ds(url)
        # ERDDAP longitude usually -180..180
        lonx = lon if ds["longitude"].min() < 0 else to_0_360(lon)
        t0 = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        u = nearest(ds["u_current"], "time", t0, "latitude", lat, "longitude", lonx).values
        v = nearest(ds["v_current"], "time", t0, "latitude", lat, "longitude", lonx).values
        spd_ms = float(np.hypot(u, v))
        return {
            "source_current": url,
            "current_speed_kts": spd_ms * KTS_PER_MS,
            "current_dir_to_degT": current_dir_to(float(u), float(v)),
        }
    except Exception as e:
        return {"error_current": str(e)}

# =========================
# UI: choose Single Point or File Upload
# =========================
mode = st.radio("Choose input mode", ["Single point", "Upload file"], horizontal=True)

def color_for_wind(ws):
    try:
        if ws < 16.0:
            return "#1ea21e"  # green
        elif ws > 25.0:
            return "#cc1f1f"  # red
        else:
            return "#f0a202"  # orange
    except Exception:
        return "#666666"

def render_map(df_points, ts_col_name="timestamp"):
    if df_points.empty:
        st.info("No points to map yet.")
        return
    clat = float(df_points["lat"].mean())
    clon = float(df_points["lon"].mean())
    m = folium.Map(location=[clat, clon], zoom_start=3, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        attr="Map data: © OpenSeaMap contributors",
        name="OpenSeaMap (Seamarks)",
        overlay=True,
        control=True,
    ).add_to(m)
    for _, r in df_points.iterrows():
        ws = r.get("wind_speed_kts", np.nan)
        wd = r.get("wind_dir_from_degT", np.nan)
        hs = r.get("sig_wave_hs_m", np.nan)
        hdir = r.get("sig_wave_dir_degT", np.nan)
        wvh = r.get("wave_hgt_m", np.nan)
        wvdir = r.get("wave_dir_degT", np.nan)
        swh = r.get("swell_hgt_m", np.nan)
        swdir = r.get("swell_dir_degT", np.nan)
        curk = r.get("current_speed_kts", np.nan)
        curdir = r.get("current_dir_to_degT", np.nan)
        tooltip = (
            f"<b>{r[ts_col_name]}</b><br>"
            f"Wind: {ws:.1f} kt from {wd:.0f}°T<br>"
            f"SigWave: {hs:.1f} m from {hdir:.0f}°T<br>"
            f"Wind‑wave: {wvh:.1f} m from {wvdir:.0f}°T<br>"
            f"Swell: {swh:.1f} m from {swdir:.0f}°T<br>"
            f"Current: {curk:.2f} kt to {curdir:.0f}°T"
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6, color=color_for_wind(ws), fill=True, fill_opacity=0.9, weight=1,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, returned_objects=[])

if mode == "Single point":
    with st.form("single"):
        c1, c2, c3 = st.columns([2,1,1])
        ts_str = c1.text_input("Timestamp (UTC, YYYY-MM-DD HH:MM)", "2025-07-10 06:00")
        lat = c2.number_input("Latitude (°)", value=24.5, format="%.6f")
        lon = c3.number_input("Longitude (°; −180..180 or 0..360)", value=54.4, format="%.6f")
        submitted = st.form_submit_button("Get met‑ocean")
    if submitted:
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except ValueError:
            st.error("Use format YYYY-MM-DD HH:MM (UTC).")
            st.stop()
        with st.spinner("Fetching…"):
            wind = get_wind(ts, lat, lon)
            wave = get_waves(ts, lat, lon)
            cur = get_currents(ts, lat, lon)
        rec = {"timestamp": ts_str, "lat": lat, "lon": lon}
        rec.update(wind); rec.update(wave); rec.update(cur)
        df_single = pd.DataFrame([rec])
        st.subheader("Result")
        st.dataframe(df_single, use_container_width=True)
        render_map(df_single)

else:
    st.markdown("**Upload CSV or Excel** with columns: `timestamp`, `lat`, `lon` (case‑insensitive). Time must be UTC.")
    example = pd.DataFrame({
        "timestamp": ["2025-07-10 06:00", "2024-11-03 12:00"],
        "lat": [24.5, 12.0],
        "lon": [54.4, 130.0],
    })
    with st.expander("Template / preview", expanded=False):
        st.dataframe(example)

    up = st.file_uploader("Choose CSV/XLSX", type=["csv", "xlsx"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up)
        else:
            df_in = pd.read_excel(up)

        df = df_in.copy()
        cols = {c.lower(): c for c in df.columns}
        ts_col = next((cols[k] for k in cols if k in ["timestamp","time","datetime"]), None)
        lat_col = next((cols[k] for k in cols if k in ["lat","latitude"]), None)
        lon_col = next((cols[k] for k in cols if k in ["lon","lng","longitude"]), None)

        if not (ts_col and lat_col and lon_col):
            st.error("Could not find timestamp/lat/lon columns. Use headers: timestamp, lat, lon.")
            st.stop()

        # parse & validate
        df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        for c in [lat_col, lon_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df["_ts"].isna().any() or df[[lat_col, lon_col]].isna().any().any():
            st.error("Some timestamps or coordinates are invalid.")
            st.stop()

        # process rows
        records = []
        prog = st.progress(0)
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            ts = row["_ts"].to_pydatetime()
            lat = float(row[lat_col]); lon = float(row[lon_col])
            wind = get_wind(ts, lat, lon)
            wave = get_waves(ts, lat, lon)
            cur = get_currents(ts, lat, lon)
            rec = {"timestamp": row[ts_col], "lat": lat, "lon": lon}
            rec.update(wind); rec.update(wave); rec.update(cur)
            # numeric fixes
            if "wind_speed_kts" in rec: rec["wind_speed_kts"] = float(rec["wind_speed_kts"])
            if "current_speed_kts" in rec: rec["current_speed_kts"] = float(rec["current_speed_kts"])
            records.append(rec)
            prog.progress(i/total)

        out = pd.DataFrame.from_records(records)
        st.subheader("Results")
        st.dataframe(out, use_container_width=True)
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name="metocean_results.csv", mime="text/csv")
        render_map(out)