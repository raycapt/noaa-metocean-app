import os
import requests
from datetime import datetime, timezone

PARAMS = [
    "windSpeed","windDirection",
    "waveHeight","waveDirection",
    "windWaveHeight","windWaveDirection",
    "swellHeight","swellDirection",
    "currentSpeed","currentDirection"
]

DEFAULT_SOURCE = "best"

class StormglassClient:
    def __init__(self, api_key: str, source: str = DEFAULT_SOURCE, timeout: int = 20):
        self.api_key = api_key
        self.source = source
        self.timeout = timeout
        self.base_url = "https://api.stormglass.io/v2/weather/point"

    def nearest_hour_window(self, dtobj: datetime):
        if dtobj.tzinfo is None:
            dtobj = dtobj.replace(tzinfo=timezone.utc)
        else:
            dtobj = dtobj.astimezone(timezone.utc)
        hour = dtobj.replace(minute=0, second=0, microsecond=0)
        iso = hour.isoformat().replace("+00:00","Z")
        return iso, iso

    def fetch_point(self, lat: float, lon: float, dtobj: datetime):
        headers = {"Authorization": self.api_key}
        start, end = self.nearest_hour_window(dtobj)
        params = {
            "lat": lat,
            "lng": lon,
            "params": ",".join(PARAMS),
            "start": start,
            "end": end,
        }
        if self.source and self.source != "best":
            params["source"] = self.source

        r = requests.get(self.base_url, headers=headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def extract_values(self, payload: dict):
        try:
            hour = payload["hours"][0]
        except Exception:
            return {}
        def get_param(p):
            node = hour.get(p)
            if isinstance(node, dict):
                if "sg" in node and node["sg"] is not None:
                    return node["sg"]
                for _, v in node.items():
                    if v is not None:
                        return v
                return None
            return node
        out = {p: get_param(p) for p in PARAMS}
        out["iso_time"] = hour.get("time")
        return out
