import streamlit as st
import pandas as pd
import numpy as np
import requests, math
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone

st.set_page_config(page_title="Global Met-Ocean @ Points (2024→Now)", layout="wide")
st.title("Global Wind · Waves · Swell · Currents @ Positions (NOAA/WW3)")
st.caption("Enter a timestamp + position or upload CSV/XLSX (timestamp, lat, lon). "
           "Waves at the requested time (hourly WW3). Winds & currents are daily, evaluated at 00:00Z of that date. "
           "Wind/Swell/Wave **directions are FROM** (°T). **Current direction is GOING TO** (°T). "
           "Wind speed shown in knots; wave heights in meters.")

KTS_PER_MS = 1.9438444924574

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "metocean-streamlit-json/1.0 (+https://github.com/)",
        "Accept": "application/json,text/*,*/*",
    })
    retry = Retry(
        total=4, connect=4, read=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

S = _session()

def erddap_point_json(base, dataset, var_exprs):
    url = f"{base}/griddap/{dataset}.json?{','.join(var_exprs)}"
    r = S.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = js.get("table", {}).get("rows", [])
    if not rows:
        return None, url
    vals = rows[0]
    if not isinstance(vals, list):
        vals = [vals]
    return vals, url

def to_iso_utc(ts: datetime):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")

def to_daily_iso(ts: datetime):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")

def lon_to_m180_180(lon):
    if lon > 180:
        return ((lon + 180) % 360) - 180
    if lon < -180:
        return ((lon + 180) % 360) - 180
    return lon

def fetch_waves(time_utc: datetime, lat: float, lon: float):
    t_iso = to_iso_utc(time_utc)
    base = "https://pae-paha.pacioos.hawaii.edu/erddap"
    ds = "ww3_global"
    q = [
        f"Thgt[({t_iso})][(0.0)][({lat})][({lon})]",
        f"Tdir[({t_iso})][(0.0)][({lat})][({lon})]",
        f"whgt[({t_iso})][(0.0)][({lat})][({lon})]",
        f"wdir[({t_iso})][(0.0)][({lat})][({lon})]",
        f"shgt[({t_iso})][(0.0)][({lat})][({lon})]",
        f"sdir[({t_iso})][(0.0)][({lat})][({lon})]",
    ]
    vals, url = erddap_point_json(base, ds, q)
    if vals is None:
        return {"error_wave": "No WW3 value at/near time/point", "source_wave": url}
    Thgt, Tdir, whgt, wdir, shgt, sdir = [None if v is None else float(v) for v in vals[-6:]]
    return {
        "source_wave": url,
        "sig_wave_hs_m": Thgt, "sig_wave_dir_degT": Tdir,
        "wave_hgt_m": whgt,     "wave_dir_degT": wdir,
        "swell_hgt_m": shgt,    "swell_dir_degT": sdir,
    }

def fetch_wind_daily(time_utc: datetime, lat: float, lon: float):
    t_iso = to_daily_iso(time_utc)
    lon_m = lon_to_m180_180(lon)
    base = "https://coastwatch.noaa.gov/erddap"
    ds = "noaacwBlendedWindsDaily"
    q = [
        f"windspeed[({t_iso})][(10.0)][({lat})][({lon_m})]",
        f"u_wind[({t_iso})][(10.0)][({lat})][({lon_m})]",
        f"v_wind[({t_iso})][(10.0)][({lat})][({lon_m})]",
    ]
    vals, url = erddap_point_json(base, ds, q)
    if vals is None:
        return {"error_wind": "No wind daily value at/near time/point", "source_wind": url}
    ws_ms, uw, vw = [None if v is None else float(v) for v in vals[-3:]]
    if ws_ms is None or uw is None or vw is None:
        return {"error_wind": "Wind value missing", "source_wind": url}
    ws_kts = ws_ms * KTS_PER_MS
    wdir_from = (270.0 - math.degrees(math.atan2(vw, uw))) % 360.0
    return {
        "source_wind": url,
        "wind_speed_kts": ws_kts,
        "wind_dir_from_degT": wdir_from,
        "u_wind_ms": uw, "v_wind_ms": vw,
    }

def fetch_currents_daily(time_utc: datetime, lat: float, lon: float):
    t_iso = to_daily_iso(time_utc)
    lon_m = lon_to_m180_180(lon)
    base = "https://coastwatch.noaa.gov/erddap"
    ds = "noaacwBLENDEDNRTcurrentsDaily"
    q = [
        f"u_current[({t_iso})][({lat})][({lon_m})]",
        f"v_current[({t_iso})][({lat})][({lon_m})]",
    ]
    try:
        vals, url = erddap_point_json(base, ds, q)
        if vals is not None:
            u, v = [None if v is None else float(v) for v in vals[-2:]]
            if u is not None and v is not None:
                spd_ms = float(math.hypot(u, v))
                dir_to = ( math.degrees(math.atan2(v, u)) ) % 360.0
                return {
                    "source_current": url,
                    "current_speed_kts": spd_ms * KTS_PER_MS,
                    "current_dir_to_degT": dir_to,
                    "u_current_ms": u, "v_current_ms": v,
                }
    except Exception:
        pass
    base = "https://coastwatch.pfeg.noaa.gov/erddap"
    ds = "oscar_vel"
    q = [
        f"u[({t_iso})][({lat})][({lon_m})]",
        f"v[({t_iso})][({lat})][({lon_m})]",
    ]
    try:
        vals, url = erddap_point_json(base, ds, q)
        if vals is not None:
            u, v = [None if v is None else float(v) for v in vals[-2:]]
            if u is not None and v is not None:
                spd_ms = float(math.hypot(u, v))
                dir_to = ( math.degrees(math.atan2(v, u)) ) % 360.0
                return {
                    "source_current": url,
                    "current_speed_kts": spd_ms * KTS_PER_MS,
                    "current_dir_to_degT": dir_to,
                    "u_current_ms": u, "v_current_ms": v,
                }
    except Exception:
        pass
    return {"error_current": "No surface current found (both datasets)"}

def wind_color(ws_kts):
    try:
        if ws_kts < 16:  return "#16a34a"
        if ws_kts <= 25: return "#f59e0b"
        return "#dc2626"
    except Exception:
        return "#6b7280"

def render_map(df):
    import folium
    from streamlit_folium import st_folium
    if df.empty:
        st.info("No points to display"); return
    clat = float(df["lat"].mean()); clon = float(df["lon"].mean())
    m = folium.Map(location=[clat, clon], zoom_start=3, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        attr="Map data: © OpenSeaMap contributors",
        name="OpenSeaMap (Seamarks)",
        overlay=True, control=True,
    ).add_to(m)
    for _, r in df.iterrows():
        ws = r.get("wind_speed_kts", np.nan); wd = r.get("wind_dir_from_degT", np.nan)
        sig = r.get("sig_wave_hs_m", np.nan); sigd = r.get("sig_wave_dir_degT", np.nan)
        wvh = r.get("wave_hgt_m", np.nan); wvd = r.get("wave_dir_degT", np.nan)
        swh = r.get("swell_hgt_m", np.nan); swd = r.get("swell_dir_degT", np.nan)
        curk = r.get("current_speed_kts", np.nan); curd = r.get("current_dir_to_degT", np.nan)
        tooltip = (
            f"<b>{r['timestamp']}</b><br>"
            f"Wind: {ws:.1f} kt from {wd:.0f}°T<br>"
            f"SigWave: {sig:.1f} m from {sigd:.0f}°T<br>"
            f"Wind-sea: {wvh:.1f} m from {wvd:.0f}°T<br>"
            f"Swell: {swh:.1f} m from {swd:.0f}°T<br>"
            f"Current: {curk:.2f} kt to {curd:.0f}°T"
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6, color=wind_color(ws), fill=True, fill_opacity=0.9, weight=1,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, returned_objects=[])

mode = st.radio("Choose input", ["Single point", "Upload file"], horizontal=True)

def compute_row(ts_str, lat, lon):
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        return {"timestamp": ts_str, "lat": lat, "lon": lon, "error": "Invalid timestamp (use YYYY-MM-DD HH:MM UTC)"}
    waves = fetch_waves(ts, lat, lon)
    wind  = fetch_wind_daily(ts, lat, lon)
    curr  = fetch_currents_daily(ts, lat, lon)
    out = {"timestamp": ts_str, "lat": float(lat), "lon": float(lon)}
    out.update(waves); out.update(wind); out.update(curr)
    return out

if mode == "Single point":
    with st.form("single"):
        c1, c2, c3 = st.columns([2,1,1])
        ts_str = c1.text_input("Timestamp (UTC, YYYY-MM-DD HH:MM)", "2025-07-10 06:00")
        lat = c2.number_input("Latitude (°)", value=-6.0, format="%.6f")
        lon = c3.number_input("Longitude (° East positive; −180..180)", value=55.0, format="%.6f")
        sub = st.form_submit_button("Get data")
    if sub:
        with st.spinner("Fetching from ERDDAP…"):
            rec = compute_row(ts_str, lat, lon)
        df = pd.DataFrame([rec])
        st.subheader("Result")
        st.dataframe(df, use_container_width=True)
        render_map(df)
else:
    st.markdown("Upload **CSV/XLSX** with columns: `timestamp` (UTC `YYYY-MM-DD HH:MM`), `lat`, `lon`.")
    up = st.file_uploader("Choose file", type=["csv", "xlsx"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up)
        else:
            df_in = pd.read_excel(up)
        cols = {c.lower().strip(): c for c in df_in.columns}
        ts_col = cols.get("timestamp") or cols.get("time") or cols.get("datetime")
        lat_col = cols.get("lat") or cols.get("latitude")
        lon_col = cols.get("lon") or cols.get("longitude") or cols.get("lng")
        if not (ts_col and lat_col and lon_col):
            st.error("Need headers: timestamp, lat, lon"); st.stop()
        rows = []
        prog = st.progress(0); total = len(df_in)
        for i, (_, r) in enumerate(df_in.iterrows(), start=1):
            ts_str = str(r[ts_col])
            try:
                lat = float(r[lat_col]); lon = float(r[lon_col])
            except Exception:
                rows.append({"timestamp": ts_str, "lat": r[lat_col], "lon": r[lon_col], "error": "Invalid lat/lon"})
                prog.progress(i/total); continue
            rows.append(compute_row(ts_str, lat, lon))
            prog.progress(i/total)
        out = pd.DataFrame(rows)
        st.subheader("Results")
        st.dataframe(out, use_container_width=True)
        st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="metocean_results.csv", mime="text/csv")
        render_map(out)
