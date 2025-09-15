import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Met-Ocean at Points · 2024→today", layout="wide")
st.title("Historical NOAA Wind · Wave · Current at Positions")
st.caption("Upload a CSV/XLSX OR enter a single timestamp/position. I’ll fetch wind (10 m), waves (WW3), and surface currents.\n"
           "Directions: Wind FROM (°T), currents TO (°T). Wave/swell directions are FROM (°T).")

# =========================
# Constants / helpers
# =========================
KTS_PER_MS = 1.9438444924574

@st.cache_data(show_spinner=False)
def open_ds(url: str):
    # Prefer pydap on Streamlit Cloud for OPeNDAP/ERDDAP
    try:
        return xr.open_dataset(url, engine="pydap")
    except Exception:
        return xr.open_dataset(url)

@lru_cache(maxsize=1024)
def gfswave_url_for(ts: datetime):
    cyc = f"{(ts.hour // 6) * 6:02d}"
    return f"https://nomads.ncep.noaa.gov:9090/dods/wave/gfswave/{ts:%Y%m%d}/gfswave.global.0p25_{cyc}z"

@lru_cache(maxsize=1024)
def psl_u10_url(year: int):
    return f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface_gauss/uwnd.10m.gauss.{year}.nc"

@lru_cache(maxsize=1024)
def psl_v10_url(year: int):
    return f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface_gauss/vwnd.10m.gauss.{year}.nc"

@lru_cache(maxsize=1)
def pacioos_ww3_url():
    return "https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ww3_global/WaveWatch_III_Global_Wave_Model_best.ncd"

@lru_cache(maxsize=1)
def coastwatch_currents_url():
    # ERDDAP OPeNDAP endpoint (.dods) for pydap
    return "https://coastwatch.noaa.gov/erddap/griddap/noaacwBLENDEDNRTcurrentsDaily.dods"

def to_0_360(lon):
    return lon if lon >= 0 else lon + 360

def to_naive_utc(ts: datetime):
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(timezone.utc).replace(tzinfo=None)

def wind_from_uv(u, v):
    speed_ms = np.hypot(u, v)
    deg_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return float(speed_ms), float(deg_from)

def current_dir_to(u, v):
    return float((90.0 - np.degrees(np.arctan2(v, u))) % 360.0)

def guess_axes(da):
    dims = list(da.dims)
    t = next((d for d in dims if d in ("time","valid_time")), dims[0])
    y = next((d for d in dims if d in ("lat","latitude","y")), dims[1 if len(dims)>1 else 0])
    x = next((d for d in dims if d in ("lon","longitude","x")), dims[2 if len(dims)>2 else -1])
    return t, y, x

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _finite_mask(arr):
    # Handle masked arrays from pydap
    if isinstance(arr, np.ma.MaskedArray):
        mask = ~arr.mask
        data = arr.filled(np.nan)
        return data, mask
    else:
        return arr, np.isfinite(arr)

def nearest_ocean_value(da, ts, lat, lon, max_deg=0.75, expand_deg=2.0):
    """Select nearest point; if NaN/masked, search a box (±max_deg), else expand to ±expand_deg."""
    t, y, x = guess_axes(da)
    lon_vec = da.coords[x].values
    lon_req = lon
    if np.min(lon_vec) >= 0 and lon < 0:
        lon_req = lon + 360
    if np.max(lon_vec) > 180 and lon_req < 0:
        lon_req = lon_req + 360

    # 1) direct
    try:
        sel = da.sel({t: ts, y: lat, x: lon_req}, method="nearest")
        val = sel.values
        data, mask = _finite_mask(val)
        valf = float(np.array(data).squeeze()) if np.size(data) == 1 else np.nan
        if np.isfinite(valf):
            return valf, float(sel.coords[y]), float(sel.coords[x])
    except Exception:
        pass

    def search_box(rad):
        box = da.sel({t: ts, y: slice(lat-rad, lat+rad), x: slice(lon_req-rad, lon_req+rad)})
        if box.size == 0:
            return np.nan, lat, lon_req
        yy = box.coords[y].values
        xx = box.coords[x].values
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        dist = haversine_km(lat, lon_req, Y, X)

        arr = np.array(box.values)
        data, mask = _finite_mask(arr)
        if np.ndim(data) >= 2:
            # last two dims are y,x
            d2 = dist.copy()
            d2[~mask] = np.inf
            if not np.isfinite(d2).any():
                return np.nan, lat, lon_req
            iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
            val = float(data[iy, ix])
            return val, float(yy[iy]), float(xx[ix])
        else:
            return np.nan, lat, lon_req

    # try small box then bigger
    val, y_used, x_used = search_box(max_deg)
    if not np.isfinite(val):
        val, y_used, x_used = search_box(expand_deg)
    return val, y_used, x_used

def find_var(ds, *cands):
    """Find first variable whose lowercase name contains any of cands (strings)."""
    cands = [c.lower() for c in cands]
    for k in ds.data_vars:
        kl = k.lower()
        if any(c in kl for c in cands):
            return k
    raise KeyError("No variable matched: " + ",".join(cands))

# =========================
# Data fetchers
# =========================
def get_wind(ts: datetime, lat: float, lon: float):
    tsn = to_naive_utc(ts)
    today = datetime.now(timezone.utc).date()
    recent_cut = today - timedelta(days=6)
    lon0360 = to_0_360(lon)

    if ts.date() >= recent_cut:
        try:
            url = gfswave_url_for(ts)
            ds = open_ds(url)
            wspd_ms, y_used, x_used = nearest_ocean_value(ds["windsfc"], tsn, lat, lon0360)
            wdir, _, _ = nearest_ocean_value(ds["wdirsfc"], tsn, y_used, x_used)
            return {"source_wind": url, "wind_speed_kts": float(wspd_ms) * KTS_PER_MS, "wind_dir_from_degT": float(wdir)}
        except Exception:
            pass

    # PSL fallback (global, continuous)
    try:
        u = open_ds(psl_u10_url(ts.year))
        v = open_ds(psl_v10_url(ts.year))
        uval, y_used, x_used = nearest_ocean_value(u["uwnd"], tsn, lat, lon0360)
        vval, _, _ = nearest_ocean_value(v["vwnd"], tsn, y_used, x_used)
        spd_ms, dir_from = wind_from_uv(float(uval), float(vval))
        return {"source_wind": psl_u10_url(ts.year), "wind_speed_kts": spd_ms * KTS_PER_MS, "wind_dir_from_degT": dir_from}
    except Exception as e2:
        return {"error_wind": f"{e2}"}

def get_waves(ts: datetime, lat: float, lon: float):
    tsn = to_naive_utc(ts)
    today = datetime.now(timezone.utc).date()
    recent_cut = today - timedelta(days=6)
    try:
        if ts.date() >= recent_cut:
            url = gfswave_url_for(ts)
            ds = open_ds(url)
            lon0360 = to_0_360(lon)
            hs_val, y_used, x_used = nearest_ocean_value(ds["htsgwsfc"], tsn, lat, lon0360)
            dir_sig_val, _, _      = nearest_ocean_value(ds["dirpwsfc"], tsn, y_used, x_used)
            wvh_val, _, _          = nearest_ocean_value(ds["wvhgtsfc"], tsn, y_used, x_used)
            wvdir_val, _, _        = nearest_ocean_value(ds["wvdirsfc"], tsn, y_used, x_used)
            swh_val, _, _          = nearest_ocean_value(ds["swell_1"], tsn, y_used, x_used)
            swdir_val, _, _        = nearest_ocean_value(ds["swdir_1"], tsn, y_used, x_used)
            return {
                "source_wave": url,
                "sampled_lat": y_used, "sampled_lon": x_used,
                "wave_hgt_m": wvh_val, "wave_dir_degT": wvdir_val,
                "swell_hgt_m": swh_val, "swell_dir_degT": swdir_val,
                "sig_wave_hs_m": hs_val, "sig_wave_dir_degT": dir_sig_val,
            }
        else:
            url = pacioos_ww3_url()
            ds = open_ds(url)
            var_hsig  = find_var(ds, "thgt", "hs", "significant wave height")
            var_dirsg = find_var(ds, "tdir", "primary wave dir", "dirpw")
            var_wvh   = find_var(ds, "whgt", "wind wave height", "sea height", "wind_sea")
            var_wvdir = find_var(ds, "wdir", "wind wave direction", "sea direction")
            var_swh   = find_var(ds, "shgt", "swell height")
            var_swdir = find_var(ds, "sdir", "swell direction")

            hs_val, y_used, x_used = nearest_ocean_value(ds[var_hsig], tsn, lat, lon)
            dir_sig_val, _, _      = nearest_ocean_value(ds[var_dirsg], tsn, y_used, x_used)
            wvh_val, _, _          = nearest_ocean_value(ds[var_wvh], tsn, y_used, x_used)
            wvdir_val, _, _        = nearest_ocean_value(ds[var_wvdir], tsn, y_used, x_used)
            swh_val, _, _          = nearest_ocean_value(ds[var_swh], tsn, y_used, x_used)
            swdir_val, _, _        = nearest_ocean_value(ds[var_swdir], tsn, y_used, x_used)

            if not np.isfinite(wvh_val) and np.isfinite(hs_val) and np.isfinite(swh_val):
                wvh_val = max(0.0, (hs_val**2 - swh_val**2))**0.5

            return {
                "source_wave": url,
                "sampled_lat": y_used, "sampled_lon": x_used,
                "wave_hgt_m": wvh_val, "wave_dir_degT": wvdir_val,
                "swell_hgt_m": swh_val, "swell_dir_degT": swdir_val,
                "sig_wave_hs_m": hs_val, "sig_wave_dir_degT": dir_sig_val,
            }
    except Exception as e:
        return {"error_wave": f"{e}"}

def get_currents(ts: datetime, lat: float, lon: float):
    tsn = to_naive_utc(ts)
    try:
        url = coastwatch_currents_url()
        ds = open_ds(url)
        # Auto-detect variable names across ERDDAP flavors
        try:
            var_u = find_var(ds, "u_current", "u sea", "eastward", "eastward_sea_water_velocity", "u")
            var_v = find_var(ds, "v_current", "v sea", "northward", "northward_sea_water_velocity", "v")
        except Exception:
            var_u, var_v = "u_current", "v_current"
        # daily; sample nearest valid ocean cell (expand up to 2° if needed)
        t0 = tsn.replace(hour=0, minute=0, second=0, microsecond=0)
        u_val, y_used, x_used = nearest_ocean_value(ds[var_u], t0, lat, lon, max_deg=0.75, expand_deg=2.0)
        v_val, _, _           = nearest_ocean_value(ds[var_v], t0, y_used, x_used, max_deg=0.75, expand_deg=2.0)
        if not (np.isfinite(u_val) and np.isfinite(v_val)):
            return {"source_current": url, "sampled_lat": y_used, "sampled_lon": x_used,
                    "current_speed_kts": np.nan, "current_dir_to_degT": np.nan}
        spd_ms = float(np.hypot(u_val, v_val))
        return {"source_current": url, "sampled_lat": y_used, "sampled_lon": x_used,
                "current_speed_kts": spd_ms * KTS_PER_MS, "current_dir_to_degT": current_dir_to(u_val, v_val)}
    except Exception as e:
        return {"error_current": f"{e}"}

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
        sampled_lat = r.get("sampled_lat", r["lat"])
        sampled_lon = r.get("sampled_lon", r["lon"])
        tooltip = (
            f"<b>{r[ts_col_name]}</b><br>"
            f"Wind: {ws:.1f} kt from {wd:.0f}°T<br>"
            f"SigWave: {hs:.1f} m from {hdir:.0f}°T<br>"
            f"Wind-wave: {wvh:.1f} m from {wvdir:.0f}°T<br>"
            f"Swell: {swh:.1f} m from {swdir:.0f}°T<br>"
            f"Current: {curk:.2f} kt to {curdir:.0f}°T<br>"
            f"<i>Sampled @ {sampled_lat:.3f}, {sampled_lon:.3f}</i>"
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6, color=color_for_wind(ws), fill=True, fill_opacity=0.9, weight=1,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, returned_objects=[])

# ---- Single point mode
if mode == "Single point":
    with st.form("single"):
        c1, c2, c3 = st.columns([2,1,1])
        ts_str = c1.text_input("Timestamp (UTC, YYYY-MM-DD HH:MM)", "2025-07-10 06:00")
        lat = c2.number_input("Latitude (°)", value=24.5, format="%.6f")
        lon = c3.number_input("Longitude (°; −180..180 or 0..360)", value=54.4, format="%.6f")
        submitted = st.form_submit_button("Get met-ocean")
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

# ---- Upload mode
else:
    st.markdown("**Upload CSV or Excel** with columns: `timestamp`, `lat`, `lon` (case-insensitive). Time must be UTC.")
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
