import os
import io
import folium
import streamlit as st
import pandas as pd
from utils import to_knots, normalize_input_df, wind_color
from stormglass_client import StormglassClient, DEFAULT_SOURCE

st.set_page_config(page_title="Nautical Weather Map", page_icon="🌊", layout="wide")

st.title("🌊 Nautical Weather Map — Stormglass")
st.caption("Enter one position/time or upload many, then visualize wind, **Significant wave (Hs)**, **Wind wave**, swell, and currents on a nautical chart.")

with st.sidebar:
    st.header("Settings")
    source = st.selectbox(
        "Stormglass source (try different ones if data missing)",
        options=["best","noaa","icon","meteo"],
        index=0
    )
    st.markdown("""**Wind speed color thresholds (knots)**  
- `< 16` = green  
- `16–24` = orange  
- `> 24` = red""")
    st.write("---")
    st.write("Add your API key in **Secrets** as `STORMGLASS_API_KEY`.")

api_key = st.secrets.get("STORMGLASS_API_KEY") or os.getenv("STORMGLASS_API_KEY")
if not api_key:
    st.warning("No Stormglass API key found. Set `STORMGLASS_API_KEY` in Streamlit secrets or environment variables.", icon="⚠️")

client = StormglassClient(api_key=api_key or "DUMMY", source=source)

st.subheader("1) Input positions & timestamps")
tab_single, tab_bulk = st.tabs(["Single point", "Bulk upload CSV/XLSX"])

with tab_single:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        ts = st.text_input("Timestamp (UTC) e.g., `2025-09-20 06:30` or ISO", value="2025-09-20 06:00")
    with c2:
        lat = st.number_input("Latitude", value=0.0, format="%.6f")
    with c3:
        lon = st.number_input("Longitude", value=0.0, format="%.6f")
    do_single = st.button("Fetch single point")

with tab_bulk:
    uploaded = st.file_uploader("Upload CSV/XLSX with columns: timestamp, lat, lon", type=["csv","xlsx"])
    do_bulk = st.button("Fetch uploaded points")

@st.cache_data(show_spinner=False)
def _fetch_one(_lat, _lon, _ts_str, _client: StormglassClient):
    parsed = pd.to_datetime(_ts_str, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    try:
        payload = _client.fetch_point(float(_lat), float(_lon), parsed.to_pydatetime())
        values = _client.extract_values(payload)
        return values
    except Exception as e:
        return {"error": str(e)}

def enrich_df(df_in: pd.DataFrame, client: StormglassClient):
    rows = []
    for _, r in df_in.iterrows():
        res = _fetch_one(r["lat"], r["lon"], r["parsed_ts"].isoformat(), client)
        if res is None:
            rows.append({})
            continue
        rec = {
            "timestamp_utc": r["parsed_ts"],
            "lat": r["lat"],
            "lon": r["lon"],
        }
        rec.update(res or {})
        rows.append(rec)
    out = pd.DataFrame(rows)

    # Convert speeds to knots
    if "windSpeed" in out:
        out["windSpeed_kt"] = out["windSpeed"].apply(to_knots)
    if "currentSpeed" in out:
        out["currentSpeed_kt"] = out["currentSpeed"].apply(to_knots)

    # Rename for clarity (explicit fields)
    rename_map = {
        "windDirection": "windDir_deg_from",
        "waveHeight": "sigWaveHeight_m",
        "waveDirection": "sigWaveDir_deg_from",
        "windWaveHeight": "windWaveHeight_m",
        "windWaveDirection": "windWaveDir_deg_from",
        "swellDirection": "swellDir_deg_from",
        "swellHeight": "swellHeight_m",
        "currentDirection": "currentDir_deg_to",
    }
    out.rename(columns=rename_map, inplace=True)

    # Order columns
    preferred_cols = [
        "timestamp_utc","iso_time","lat","lon",
        "windSpeed_kt","windDir_deg_from",
        "sigWaveHeight_m","sigWaveDir_deg_from",
        "windWaveHeight_m","windWaveDir_deg_from",
        "swellHeight_m","swellDir_deg_from",
        "currentSpeed_kt","currentDir_deg_to"
    ]
    existing = [c for c in preferred_cols if c in out.columns]
    others = [c for c in out.columns if c not in existing]
    out = out[existing + others]
    return out

def make_map(df_points: pd.DataFrame):
    if df_points.empty:
        return None
    center = [df_points["lat"].mean(), df_points["lon"].mean()]
    m = folium.Map(location=center, zoom_start=3, tiles="OpenStreetMap", control_scale=True)

    # OpenSeaMap nautical overlay
    folium.TileLayer(
        tiles="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        attr="Map data: © OpenSeaMap contributors",
        name="OpenSeaMap (Nautical)",
        overlay=True,
        control=True
    ).add_to(m)

    for _, r in df_points.iterrows():
        wind_kt = r.get("windSpeed_kt")
        color = wind_color(wind_kt if wind_kt is not None else float("nan"))

        tt = folium.Tooltip(
            f"""
<b>Time (UTC):</b> {r.get('timestamp_utc') or ''}<br>
<b>Lat/Lon:</b> {r.get('lat'):.4f}, {r.get('lon'):.4f}<br>
<b>Wind:</b> {r.get('windSpeed_kt','')} kt @ {r.get('windDir_deg_from','')}° (from)<br>
<b>Significant wave (Hs):</b> {r.get('sigWaveHeight_m','')} m @ {r.get('sigWaveDir_deg_from','')}° (from)<br>
<b>Wind wave:</b> {r.get('windWaveHeight_m','')} m @ {r.get('windWaveDir_deg_from','')}° (from)<br>
<b>Swell:</b> {r.get('swellHeight_m','')} m @ {r.get('swellDir_deg_from','')}° (from)<br>
<b>Current:</b> {r.get('currentSpeed_kt','')} kt @ {r.get('currentDir_deg_to','')}° (to)
""",
            sticky=True
        )

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=1
        ).add_child(tt).add_to(m)

    folium.LayerControl().add_to(m)
    return m

result_df = None

if do_single:
    df = pd.DataFrame([{"timestamp": ts, "lat": lat, "lon": lon}])
    try:
        df_norm = df.rename(columns={"timestamp":"timestamp","lat":"lat","lon":"lon"})
        df_norm["parsed_ts"] = pd.to_datetime(df_norm["timestamp"], utc=True, errors="coerce")
        df_norm = df_norm.dropna(subset=["parsed_ts"])
        result_df = enrich_df(df_norm, client)
    except Exception as e:
        st.error(f"Failed to parse/fetch: {e}")

if do_bulk and uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            df_in = pd.read_excel(uploaded)
        df_norm = normalize_input_df(df_in)
        result_df = enrich_df(df_norm, client)
    except Exception as e:
        st.error(f"Upload or processing error: {e}")

if isinstance(result_df, pd.DataFrame) and not result_df.empty:
    st.subheader("2) Results")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="nautical_weather_results.csv", mime="text/csv")

    st.subheader("3) Map")
    m = make_map(result_df)
    if m:
        from streamlit.components.v1 import html as st_html
        st_html(m.get_root().render(), height=600, scrolling=False)
    else:
        st.info("No map to display yet.")
else:
    st.info("Enter a point or upload a file, then click **Fetch**.")
