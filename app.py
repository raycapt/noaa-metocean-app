import streamlit as st
import pandas as pd
import numpy as np
import requests, math, os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="Global Met-Ocean @ Points (2024→Now)", layout="wide")
st.title("Global Wind · Waves · Swell · Currents @ Positions")
st.caption("v11 — Stormglass + NOAA/WW3/ERA5 fallbacks. "
           "Wave/swell/wind dirs are FROM (°T). Currents are TO (°T). Wind/current speeds in knots.")

KTS_PER_MS = 1.9438444924574

# ---------- Config / Secrets ----------
def get_sg_api_key():
    # Prefer Streamlit secrets
    try:
        k = st.secrets.get("STORMGLASS_API_KEY", None)
        if k: return k
    except Exception:
        pass
    # Fallback to environment variable
    k = os.environ.get("STORMGLASS_API_KEY")
    return k

def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "metocean-streamlit/1.0 (+github)"})
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s
S = _session()

def erddap_point_json(base, dataset, var_exprs, timeout=30):
    url = f"{base}/griddap/{dataset}.json?{','.join(var_exprs)}"
    http_status = 0
    try:
        r = S.get(url, timeout=timeout)
        http_status = r.status_code
        r.raise_for_status()
        js = r.json()
        rows = js.get("table", {}).get("rows", [])
        if not rows:
            return None, url, http_status
        vals = rows[0]
        if not isinstance(vals, list):
            vals = [vals]
        return vals, url, http_status
    except Exception:
        return None, url, http_status

def to_iso_utc(ts: datetime):
    if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
    else: ts = ts.astimezone(timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")

def to_daily_iso(ts: datetime):
    if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
    else: ts = ts.astimezone(timezone.utc)
    ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")

def lon_to_m180_180(lon):
    if lon > 180: return ((lon + 180) % 360) - 180
    if lon < -180: return ((lon + 180) % 360) - 180
    return lon

def lon_to_0_360(lon):
    if lon < 0: return lon + 360.0
    return lon

def erddap_nearest_time(base, dataset, t_iso_want):
    q = [f"time[({t_iso_want})]"]
    vals, url, status = erddap_point_json(base, dataset, q)
    if vals is None or not vals:
        return None, f"{base}/griddap/{dataset}.json?"+",".join(q)
    v = vals[-1]
    try:
        if isinstance(v, (int, float)):
            iso = datetime.fromtimestamp(float(v), tz=timezone.utc).isoformat().replace("+00:00","Z")
            return iso, url
        s = str(v)
        if "T" in s and s.endswith("Z"): return s, url
        return s, url
    except Exception:
        return None, url

# ---------- Stormglass integration ----------
SG_DEFAULT_SOURCES = ["noaa","meto","icon","gfs","dwd","fmi","mf","smhi","yr","bom","sg"]

def sg_pick(value_obj, sources=SG_DEFAULT_SOURCES):
    """value_obj is like {'noaa': 1.2, 'sg': 1.3}. Pick first available by source priority."""
    if value_obj is None: return None, None
    if isinstance(value_obj, (int,float)):  # already scalar
        return float(value_obj), "sg"
    if isinstance(value_obj, dict):
        for s in sources:
            if s in value_obj and value_obj[s] is not None:
                try:
                    return float(value_obj[s]), s
                except Exception:
                    continue
    return None, None

def fetch_stormglass(time_utc: datetime, lat: float, lon: float, sources=SG_DEFAULT_SOURCES):
    """Fetch wind, waves, swell, currents for the hour (Stormglass point API)."""
    api_key = get_sg_api_key()
    if not api_key:
        return {"sg_used": False, "sg_note": "No API key configured"}

    ts = time_utc.astimezone(timezone.utc) if time_utc.tzinfo else time_utc.replace(tzinfo=timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    start = f"{date_str}T00:00:00Z"
    end   = f"{date_str}T23:59:59Z"

    params = ",".join([
        "windSpeed","windDirection",
        "waveHeight","waveDirection",
        "swellHeight","swellDirection",
        "currentSpeed","currentDirection"
    ])

    headers = {"Authorization": api_key}

    url1 = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lon}&params={params}&start={start}&end={end}"
    try:
        r = S.get(url1, headers=headers, timeout=30)
        ok1 = r.ok
        js1 = r.json() if ok1 else {}
    except Exception:
        ok1 = False; js1 = {}

    need_marine = False
    for k in ("waveHeight","swellHeight","currentSpeed"):
        if ok1 and js1.get("hours"):
            if all(k not in h for h in js1["hours"]):
                need_marine = True
        else:
            need_marine = True
    js2 = {}; url2 = None
    if need_marine:
        url2 = f"https://api.stormglass.io/v2/marine/point?lat={lat}&lng={lon}&params={params}&start={start}&end={end}"
        try:
            r2 = S.get(url2, headers=headers, timeout=30)
            if r2.ok: js2 = r2.json()
        except Exception:
            js2 = {}

    def hours_by_time(obj):
        m = {}
        for h in obj.get("hours", []) if isinstance(obj, dict) else []:
            t = h.get("time"); 
            if t: m[t] = h
        return m

    h1 = hours_by_time(js1); h2 = hours_by_time(js2)

    def pick_closest_hour():
        want = ts.replace(minute=0, second=0, microsecond=0)
        candidates = set(list(h1.keys()) + list(h2.keys()))
        if not candidates: return None, None
        def tparse(s):
            try:
                return datetime.fromisoformat(s.replace("Z","+00:00"))
            except Exception:
                return None
        best = None; best_dt = None
        for s in candidates:
            dt = tparse(s); 
            if dt is None: continue
            if best_dt is None or abs((dt - want).total_seconds()) < abs((best_dt - want).total_seconds()):
                best_dt = dt; best = s
        return best, best_dt

    best_key, best_dt = pick_closest_hour()
    if not best_key:
        return {"sg_used": False, "sg_note": "Stormglass returned no hours for this day",
                "source_sg_weather": url1, "source_sg_marine": url2}

    def getp(name):
        val_obj = None
        if best_key in h1 and name in h1[best_key]: val_obj = h1[best_key][name]
        if (val_obj is None or (isinstance(val_obj, dict) and not val_obj)) and best_key in h2 and name in h2[best_key]:
            val_obj = h2[best_key][name]
        val, src = sg_pick(val_obj, sources)
        return val, src

    ws, src_ws = getp("windSpeed")
    wd, src_wd = getp("windDirection")
    wh, src_wh = getp("waveHeight")
    wd_wv, src_wd_wv = getp("waveDirection")
    sh, src_sh = getp("swellHeight")
    sd, src_sd = getp("swellDirection")
    cs, src_cs = getp("currentSpeed")
    cd, src_cd = getp("currentDirection")

    out = {"sg_used": True, "wind_time_used": best_key}
    if ws is not None:
        out["wind_speed_kts"] = float(ws) * KTS_PER_MS
        out["wind_dir_from_degT"] = float(wd) if wd is not None else None
        out["u_wind_ms"] = None; out["v_wind_ms"] = None
        out["source_wind"] = (url1 if ws is not None else url2)
        out["note_wind"] = f"Stormglass ({src_ws or 'sg'})"
    if wh is not None:
        out["wave_hgt_m"] = float(wh)
        out["wave_dir_degT"] = float(wd_wv) if wd_wv is not None else None
        out["source_wave"] = (url1 if wh is not None else url2)
        out["note_wave"] = f"Stormglass ({src_wh or 'sg'})"
    if sh is not None:
        out["swell_hgt_m"] = float(sh)
        out["swell_dir_degT"] = float(sd) if sd is not None else None
    if cs is not None:
        out["current_speed_kts"] = float(cs) * KTS_PER_MS
        out["current_dir_to_degT"] = float(cd) if cd is not None else None
        out["source_current"] = (url1 if cs is not None else url2)
        out["note_current"] = f"Stormglass ({src_cs or 'sg'})"
    if "sig_wave_hs_m" not in out and ("wave_hgt_m" in out):
        out["sig_wave_hs_m"] = out["wave_hgt_m"]
        out["sig_wave_dir_degT"] = out.get("wave_dir_degT")
    out["source_sg_weather"] = url1
    if url2: out["source_sg_marine"] = url2
    return out

# -------- Waves (PacIOOS -> NOAA NWW3 fallback) --------
def fetch_waves(time_utc: datetime, lat: float, lon: float):
    t_iso = to_iso_utc(time_utc)
    base = "https://pae-paha.pacioos.hawaii.edu/erddap"; ds = "ww3_global"; lon_p = lon_to_0_360(lon)
    q = [f"Thgt[({t_iso})][(0.0)][({lat})][({lon_p})]",
         f"Tdir[({t_iso})][(0.0)][({lat})][({lon_p})]",
         f"whgt[({t_iso})][(0.0)][({lat})][({lon_p})]",
         f"wdir[({t_iso})][(0.0)][({lat})][({lon_p})]",
         f"shgt[({t_iso})][(0.0)][({lat})][({lon_p})]",
         f"sdir[({t_iso})][(0.0)][({lat})][({lon_p})]"]
    vals, url, status = erddap_point_json(base, ds, q)
    if vals is not None:
        Thgt, Tdir, whgt, wdir, shgt, sdir = [None if v is None else float(v) for v in vals[-6:]]
        return {"source_wave": url, "sig_wave_hs_m": Thgt, "sig_wave_dir_degT": Tdir,
                "wave_hgt_m": whgt, "wave_dir_degT": wdir, "swell_hgt_m": shgt, "swell_dir_degT": sdir}
    base2="https://coastwatch.pfeg.noaa.gov/erddap"; ds2="NWW3_Global_Best"; lon_cw=lon_to_m180_180(lon)
    got={}; url2=None
    for grp in [["htsgwsfc","dirpwsfc"],["wvhgtsfc","wvdirsfc"],["swell_1","swdir_1"]]:
        q2=[f"{v}[({t_iso})][({lat})][({lon_cw})]" for v in grp]
        vals2,url2,status2=erddap_point_json(base2, ds2, q2)
        if vals2 is not None:
            for i,vname in enumerate(grp):
                try: got[vname]=None if vals2[-len(grp)+i] is None else float(vals2[-len(grp)+i])
                except Exception: got[vname]=None
    if "htsgwsfc" in got:
        return {"source_wave": url2, "sig_wave_hs_m": got.get("htsgwsfc"), "sig_wave_dir_degT": got.get("dirpwsfc"),
                "wave_hgt_m": got.get("wvhgtsfc"), "wave_dir_degT": got.get("wvdirsfc"),
                "swell_hgt_m": got.get("swell_1"), "swell_dir_degT": got.get("swdir_1"),
                "note_wave": "PacIOOS WW3 failed; using NOAA NWW3 fallback"}
    return {"error_wave": "No WW3/NWW3 value at/near time/point"}

# -------- Wind (NOAA Science -> NOAA NRT -> ERA5) --------
def fetch_wind_daily(time_utc: datetime, lat: float, lon: float):
    base="https://coastwatch.noaa.gov/erddap"; ds="noaacwBlendedWindsDaily"
    t_iso_want=to_daily_iso(time_utc)
    t_iso_actual,t_url=erddap_nearest_time(base, ds, t_iso_want)
    if not t_iso_actual: t_iso_actual=t_iso_want
    lon_m=lon_to_0_360(lon)
    q=[f"windspeed[({t_iso_actual})][(10.0)][({lat})][({lon_m})]",
       f"u_wind[({t_iso_actual})][(10.0)][({lat})][({lon_m})]",
       f"v_wind[({t_iso_actual})][(10.0)][({lat})][({lon_m})]"]
    vals,url,status=erddap_point_json(base, ds, q)
    if vals is not None:
        ws_ms,uw,vw=[None if v is None else float(v) for v in vals[-3:]]
        if ws_ms is not None and ws_ms<=-9000: ws_ms=None
        if uw is not None and uw<=-9000: uw=None
        if vw is not None and vw<=-9000: vw=None
        if (ws_ms is None) and (uw is not None) and (vw is not None):
            ws_ms=float((uw**2+vw**2)**0.5)
        if ws_ms is not None and uw is not None and vw is not None:
            ws_kts=ws_ms*KTS_PER_MS
            wdir_from=(270.0 - math.degrees(math.atan2(vw, uw)))%360.0
            return {"source_wind": url, "wind_time_used": t_iso_actual,
                    "wind_speed_kts": ws_kts, "wind_dir_from_degT": wdir_from,
                    "u_wind_ms": uw, "v_wind_ms": vw}
    ds_nrt="noaacwBlendednrtWindsDaily"
    t_iso_actual2,t_url2=erddap_nearest_time(base, ds_nrt, t_iso_want)
    if not t_iso_actual2: t_iso_actual2=t_iso_want
    q2=[f"windspeed[({t_iso_actual2})][(10.0)][({lat})][({lon_m})]",
        f"u_wind[({t_iso_actual2})][(10.0)][({lat})][({lon_m})]",
        f"v_wind[({t_iso_actual2})][(10.0)][({lat})][({lon_m})]"]
    vals2,url2,status2=erddap_point_json(base, ds_nrt, q2)
    if vals2 is not None:
        ws_ms,uw,vw=[None if v is None else float(v) for v in vals2[-3:]]
        if ws_ms is not None and ws_ms<=-9000: ws_ms=None
        if uw is not None and uw<=-9000: uw=None
        if vw is not None and vw<=-9000: vw=None
        if (ws_ms is None) and (uw is not None) and (vw is not None):
            ws_ms=float((uw**2+vw**2)**0.5)
        if ws_ms is not None and uw is not None and vw is not None:
            ws_kts=ws_ms*KTS_PER_MS
            wdir_from=(270.0 - math.degrees(math.atan2(vw, uw)))%360.0
            return {"source_wind": url2, "note_wind": "NOAA NRT winds fallback",
                    "wind_time_used": t_iso_actual2, "wind_speed_kts": ws_kts,
                    "wind_dir_from_degT": wdir_from, "u_wind_ms": uw, "v_wind_ms": vw}
    ts = time_utc.astimezone(timezone.utc) if time_utc.tzinfo else time_utc.replace(tzinfo=timezone.utc)
    yesterday_utc = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=1)).date()
    req_date = min(ts.date(), yesterday_utc)
    date_str = req_date.strftime("%Y-%m-%d")
    hour_str = ts.strftime("%Y-%m-%dT%H:00")
    om_url = ("https://archive-api.open-meteo.com/v1/era5"
              f"?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,wind_direction_10m,wind_u_component_10m,wind_v_component_10m"
              f"&start_date={date_str}&end_date={date_str}&timezone=UTC")
    try:
        r=S.get(om_url, timeout=30); r.raise_for_status(); js=r.json()
        times = js.get("hourly", {}).get("time", [])
        if hour_str in times:
            idx = times.index(hour_str)
        else:
            from datetime import datetime as _dt
            idx = min(range(len(times)), key=lambda i: abs(_dt.fromisoformat(times[i]).timestamp() - ts.timestamp())) if times else None
        if idx is not None:
            uw = js["hourly"].get("wind_u_component_10m",[None])[idx]
            vw = js["hourly"].get("wind_v_component_10m",[None])[idx]
            ws = js["hourly"].get("wind_speed_10m",[None])[idx]
            wd = js["hourly"].get("wind_direction_10m",[None])[idx]
            if (uw is not None) and (vw is not None):
                ws = float((uw**2+vw**2)**0.5)
                wd = (270.0 - math.degrees(math.atan2(float(vw), float(uw))))%360.0
            if ws is not None and wd is not None:
                return {"source_wind": om_url, "note_wind": "Open‑Meteo ERA5 fallback",
                        "wind_time_used": times[idx] if times else hour_str,
                        "wind_speed_kts": float(ws)*KTS_PER_MS, "wind_dir_from_degT": float(wd),
                        "u_wind_ms": float(uw) if uw is not None else None,
                        "v_wind_ms": float(vw) if vw is not None else None}
    except Exception:
        pass
    return {"error_wind": f"Winds not available (science HTTP {status}; NRT HTTP {status2}); ERA5 fallback also failed",
            "source_wind_science": f"{base}/griddap/{ds}.json?" + ",".join(q),
            "source_wind_nrt": f"{base}/griddap/{ds_nrt}.json?" + ",".join(q2),
            "source_wind_era5": om_url}

# -------- Currents (CoastWatch both lon domains -> OSCAR) --------
def fetch_currents_daily(time_utc: datetime, lat: float, lon: float):
    base="https://coastwatch.noaa.gov/erddap"; ds="noaacwBLENDEDNRTcurrentsDaily"
    t_iso_want=to_daily_iso(time_utc)
    t_iso_actual,t_url=erddap_nearest_time(base, ds, t_iso_want)
    if not t_iso_actual: t_iso_actual=t_iso_want

    def try_cw(lon_val):
        q=[f"u_current[({t_iso_actual})][({lat})][({lon_val})]",
           f"v_current[({t_iso_actual})][({lat})][({lon_val})]"]
        vals,url,status=erddap_point_json(base, ds, q)
        if vals is not None:
            try: u,v=[None if v is None else float(v) for v in vals[-2:]]
            except Exception: u,v=None,None
            if u is not None and v is not None:
                spd_ms=float((u**2+v**2)**0.5); dir_to=(math.degrees(math.atan2(v,u)))%360.0
                return {"source_current": url, "current_time_used": t_iso_actual,
                        "current_speed_kts": spd_ms*KTS_PER_MS, "current_dir_to_degT": dir_to,
                        "u_current_ms": u, "v_current_ms": v}
        return None, f"{base}/griddap/{ds}.json?"+",".join(q), status

    out=try_cw(lon_to_m180_180(lon))
    if not isinstance(out, tuple): return out
    _,url1,st1=out
    out2=try_cw(lon_to_0_360(lon))
    if not isinstance(out2, tuple): return out2
    _,url2,st2=out2

    base2="https://coastwatch.pfeg.noaa.gov/erddap"; ds2="oscar_vel"
    q2=[f"u[({t_iso_actual})][({lat})][({lon_to_m180_180(lon)})]",
        f"v[({t_iso_actual})][({lat})][({lon_to_m180_180(lon)})]"]
    vals2,url3,st3=erddap_point_json(base2, ds2, q2)
    if vals2 is not None:
        try: u,v=[None if v is None else float(v) for v in vals2[-2:]]
        except Exception: u,v=None,None
        if u is not None and v is not None:
            spd_ms=float((u**2+v**2)**0.5); dir_to=(math.degrees(math.atan2(v,u)))%360.0
            return {"source_current": url3, "current_time_used": t_iso_actual,
                    "current_speed_kts": spd_ms*KTS_PER_MS, "current_dir_to_degT": dir_to,
                    "u_current_ms": u, "v_current_ms": v, "note_current": "OSCAR fallback"}
    return {"error_current": f"Currents not available (CW -180..180 HTTP {st1}; CW 0..360 HTTP {st2}; OSCAR HTTP {st3})",
            "current_time_used": t_iso_actual, "current_time_query": t_url,
            "source_current_try1": url1, "source_current_try2": url2,
            "source_current_oscar": f"{base2}/griddap/{ds2}.json?"+",".join(q2)}

def wind_color(ws_kts):
    try:
        if ws_kts < 16:  return "#16a34a"
        if ws_kts <= 25: return "#f59e0b"
        return "#dc2626"
    except Exception: return "#6b7280"

def render_map(df):
    import folium
    from streamlit_folium import st_folium
    if df.empty: st.info("No points to display"); return
    clat=float(df["lat"].mean()); clon=float(df["lon"].mean())
    m = folium.Map(location=[clat, clon], zoom_start=3, tiles="OpenStreetMap")
    folium.TileLayer(tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
                     attr="Map data: © OpenSeaMap contributors", name="OpenSeaMap (Seamarks)",
                     overlay=True, control=True).add_to(m)
    for _, r in df.iterrows():
        ws=r.get("wind_speed_kts", np.nan); wd=r.get("wind_dir_from_degT", np.nan)
        sig=r.get("sig_wave_hs_m", np.nan); sigd=r.get("sig_wave_dir_degT", np.nan)
        wvh=r.get("wave_hgt_m", np.nan); wvd=r.get("wave_dir_degT", np.nan)
        swh=r.get("swell_hgt_m", np.nan); swd=r.get("swell_dir_degT", np.nan)
        curk=r.get("current_speed_kts", np.nan); curd=r.get("current_dir_to_degT", np.nan)
        tooltip=(f"<b>{r['timestamp']}</b><br>"
                 f"Wind: {ws:.1f} kt from {wd:.0f}°T<br>"
                 f"SigWave: {sig:.1f} m from {sigd:.0f}°T<br>"
                 f"Wind-sea: {wvh:.1f} m from {wvd:.0f}°T<br>"
                 f"Swell: {swh:.1f} m from {swd:.0f}°T<br>"
                 f"Current: {curk:.2f} kt to {curd:.0f}°T")
        folium.CircleMarker(location=[float(r["lat"]), float(r["lon"])],
                            radius=6, color=wind_color(ws), fill=True, fill_opacity=0.9, weight=1,
                            tooltip=folium.Tooltip(tooltip, sticky=True)).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, returned_objects=[])

# Sidebar: Stormglass toggle & source priority
with st.sidebar:
    st.markdown("### Stormglass")
    sg_key_present = bool(get_sg_api_key())
    if sg_key_present:
        st.success("API key found in secrets/env. Stormglass will be used first.")
    else:
        st.info("No API key found. Add **STORMGLASS_API_KEY** in Streamlit **Secrets** or ENV to enable Stormglass.")
    src_order = st.text_input("Source priority (comma-separated)", ",".join(SG_DEFAULT_SOURCES))
    try:
        SG_SOURCES = [s.strip() for s in src_order.split(",") if s.strip()]
    except Exception:
        SG_SOURCES = SG_DEFAULT_SOURCES

mode = st.radio("Choose input", ["Single point", "Upload file"], horizontal=True)

def compute_row(ts_str, lat, lon):
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        return {"timestamp": ts_str, "lat": lat, "lon": lon, "error": "Invalid timestamp (use YYYY-MM-DD HH:MM UTC)"}

    # 1) Stormglass first (if key available)
    sg = fetch_stormglass(ts, lat, lon, sources=SG_SOURCES) if get_sg_api_key() else {"sg_used": False}

    # 2) For anything missing, fall back to the original providers
    waves = {} if sg.get("sig_wave_hs_m") or sg.get("wave_hgt_m") else fetch_waves(ts, lat, lon)
    wind  = {} if sg.get("wind_speed_kts") else fetch_wind_daily(ts, lat, lon)
    curr  = {} if sg.get("current_speed_kts") else fetch_currents_daily(ts, lat, lon)

    out = {"timestamp": ts_str, "lat": float(lat), "lon": float(lon)}
    for d in (sg, waves, wind, curr):
        if d: out.update(d)
    return out

if mode == "Single point":
    with st.form("single"):
        c1, c2, c3 = st.columns([2,1,1])
        ts_str = c1.text_input("Timestamp (UTC, YYYY-MM-DD HH:MM)", "2025-07-10 06:00")
        lat = c2.number_input("Latitude (°)", value=-6.0, format="%.6f")
        lon = c3.number_input("Longitude (° East positive; −180..180)", value=55.0, format="%.6f")
        sub = st.form_submit_button("Get data")
    if sub:
        with st.spinner("Fetching…"):
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
