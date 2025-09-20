
import streamlit as st
import pandas as pd
import requests
import datetime
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# -- SETTINGS --
st.set_page_config(layout="wide")
API_KEY = st.secrets.get("STORMGLASS_API", "YOUR_API_KEY_HERE")
API_URL = "https://api.stormglass.io/v2/weather/point"
PARAMS = ["windSpeed", "waveHeight", "swellHeight", "swellDirection", "swellPeriod", "currentSpeed"]

# -- HELPERS --
def fetch_weather_data(lat, lon, time_utc):
    iso_time = time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": API_KEY}
    params = {
        "lat": lat,
        "lng": lon,
        "params": ",".join(PARAMS),
        "start": iso_time,
        "end": iso_time,
        "source": "noaa"
    }
    r = requests.get(API_URL, headers=headers, params=params)
    if r.status_code != 200:
        return {"error": f"API Error {r.status_code}"}
    data = r.json().get("hours", [{}])[0]
    result = {p: data.get(p, {}).get("noaa", None) for p in PARAMS}
    return result

def is_valid_row(row):
    try:
        datetime.datetime.strptime(str(row['timestamp']), "%Y-%m-%d %H:%M:%S")
        return pd.notna(row['lat']) and pd.notna(row['lon'])
    except:
        return False

# -- INTERFACE --
st.title("🌊 Stormglass Weather Fetcher")
st.markdown("Enter up to 5 manual positions and upload Excel for bulk fetch. Hover map to view data.")

# Manual input
st.subheader("Manual Data Entry")
manual_data = []
with st.form("manual_form"):
    for i in range(5):
        cols = st.columns(3)
        ts = cols[0].text_input(f"Timestamp {i+1} (UTC, YYYY-MM-DD HH:MM:SS)", key=f"ts{i}")
        lat = cols[1].text_input(f"Latitude {i+1}", key=f"lat{i}")
        lon = cols[2].text_input(f"Longitude {i+1}", key=f"lon{i}")
        if ts and lat and lon:
            manual_data.append({"timestamp": ts, "lat": float(lat), "lon": float(lon)})
    submitted = st.form_submit_button("Fetch Weather for Manual Entries")

# Upload file
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Upload Excel with headers: timestamp, lat, lon", type=["xlsx"])
file_data = []
if uploaded_file:
    df_file = pd.read_excel(uploaded_file)
    df_file = df_file.dropna(subset=["timestamp", "lat", "lon"])
    df_file["timestamp"] = pd.to_datetime(df_file["timestamp"], errors='coerce')
    file_data = df_file.to_dict("records")

# Combine data
all_data = []
if submitted:
    all_data.extend(manual_data)
if file_data:
    all_data.extend(file_data)

# Fetch & display
if all_data:
    st.success(f"Processing {len(all_data)} positions...")
    results = []
    m = folium.Map(location=[0, 0], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(m)

    for row in all_data:
        try:
            ts = pd.to_datetime(row["timestamp"])
            lat = float(row["lat"])
            lon = float(row["lon"])
            weather = fetch_weather_data(lat, lon, ts)
            row.update(weather)
            popup = f"Lat: {lat}, Lon: {lon}<br>Time: {ts}<br>"
            popup += "<br>".join([f"{k}: {v}" for k, v in weather.items()])
            folium.Marker(location=[lat, lon], popup=popup).add_to(marker_cluster)
            results.append(row)
        except Exception as e:
            row["error"] = str(e)
            results.append(row)

    df_result = pd.DataFrame(results)
    st.subheader("📌 Weather Data Table")
    st.dataframe(df_result)
    st.download_button("Download CSV", df_result.to_csv(index=False), "weather_results.csv")

    st.subheader("🗺️ Map View")
    st_folium(m, width=1000, height=600)
