import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="NOAA Stormglass App", layout="wide")
st.title("🌊 NOAA + Stormglass Metocean Extractor")
st.caption("Upload a file with `timestamp`, `lat`, and `lon`. Format: `YYYY-MM-DD HH:MM UTC` (24H format)")

stormglass_api_key = st.secrets["STORMGLASS_API_KEY"] if "STORMGLASS_API_KEY" in st.secrets else st.text_input("Enter Stormglass API Key", type="password")

uploaded_file = st.file_uploader("Upload Excel (.xlsx) file", type="xlsx")
sample_df = pd.DataFrame({
    "timestamp": ["2025-09-01 10:00 UTC", "2025-09-03 12:00 UTC"],
    "lat": [25.2, 26.5],
    "lon": [55.3, 54.9]
})

with st.expander("📋 Sample Format"):
    st.dataframe(sample_df, use_container_width=True)

def fetch_stormglass_data(ts, lat, lon):
    url = f"https://api.stormglass.io/v2/weather/point?lat={lat}&lng={lon}&params=windSpeed,waveHeight,currentSpeed&start={ts}&end={ts}"
    headers = {"Authorization": stormglass_api_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"Stormglass API error: {response.status_code}"}
    try:
        data = response.json()
        hour_data = data.get("hours", [{}])[0]
        return {
            "windSpeed": hour_data.get("windSpeed", {}).get("sg", None),
            "waveHeight": hour_data.get("waveHeight", {}).get("sg", None),
            "currentSpeed": hour_data.get("currentSpeed", {}).get("sg", None),
            "source": "Stormglass",
        }
    except Exception as e:
        return {"error": str(e)}

if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    cols = {str(c).lower().strip(): c for c in df_in.columns}

    if not all(k in cols for k in ["timestamp", "lat", "lon"]):
        st.error("Excel must contain columns: 'timestamp', 'lat', 'lon'")
        st.stop()

    df_out = []
    for _, row in df_in.iterrows():
        try:
            ts_str = str(row[cols["timestamp"]]).replace(" UTC", "").strip()
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
            ts_unix = int(ts.replace(tzinfo=timezone.utc).timestamp())
            data = fetch_stormglass_data(ts_unix, row[cols["lat"]], row[cols["lon"]])
            row_out = row.to_dict()
            row_out.update(data)
            df_out.append(row_out)
        except Exception as e:
            row_out = row.to_dict()
            row_out["error"] = str(e)
            df_out.append(row_out)

    df_result = pd.DataFrame(df_out)
    st.success("✅ Extraction complete")
    st.dataframe(df_result, use_container_width=True)
    st.download_button("⬇️ Download Results", data=df_result.to_csv(index=False), file_name="metocean_results.csv", mime="text/csv")