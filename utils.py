import math
from dateutil import parser
import pandas as pd

KTS_PER_MPS = 1.9438444924574

def to_knots(mps):
    if mps is None:
        return None
    try:
        return float(mps) * KTS_PER_MPS
    except Exception:
        return None

def safe_parse_dt(x):
    if pd.isna(x):
        return None
    try:
        ts = pd.to_datetime(x, utc=True)
        return ts.to_pydatetime()
    except Exception:
        pass
    try:
        dt = parser.parse(str(x))
        if not dt.tzinfo:
            return dt
        return dt.astimezone(tz=None)
    except Exception:
        return None

def normalize_input_df(df):
    cols = {c.strip().lower(): c for c in df.columns}
    req = {}
    for need in ["timestamp", "lat", "lon"]:
        if need in cols:
            req[need] = cols[need]
        else:
            alts = [k for k in cols if k.replace(" ", "") == need]
            if alts:
                req[need] = cols[alts[0]]
            else:
                raise ValueError(f"Missing required column: {need}")
    out = df.copy()
    out.rename(columns={req["timestamp"]:"timestamp", req["lat"]:"lat", req["lon"]:"lon"}, inplace=True)
    out["parsed_ts"] = out["timestamp"].apply(safe_parse_dt)
    out = out[~out["parsed_ts"].isna()].copy()
    return out

def wind_color(wind_speed_kt):
    if wind_speed_kt is None or (isinstance(wind_speed_kt, float) and math.isnan(wind_speed_kt)):
        return "gray"
    if wind_speed_kt < 16:
        return "green"
    if wind_speed_kt <= 24:
        return "orange"
    return "red"
