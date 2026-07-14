import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm


def sample_flight_window(df, window_minutes=30, sampling_rate_seconds=10):
    df = df.copy()
    out = []
    window_steps = window_minutes * (60 // sampling_rate_seconds)

    for fid, group in df.groupby("flight_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) > window_steps:
            group = group.iloc[-window_steps:]
        out.append(group)

    return pd.concat(out, ignore_index=True)

def resample_flight(df, freq, max_gap_s=None):
    df = df.sort_values("timestamp").set_index("timestamp")
    
    icao = df["icao"].iloc[0]
    flight_id = df["flight_id"].iloc[0]
    type_ = df["type"].iloc[0]
    t0 = df["t0"].iloc[0] if "t0" in df.columns else None
    
    t_start = df.index.min().ceil(freq)
    t_end = df.index.max().floor(freq)
    target_idx = pd.date_range(t_start, t_end, freq=freq)
    
    if len(target_idx) == 0:
        return None

    combined_idx = df.index.union(target_idx)

    if max_gap_s is not None:
        freq_s = pd.Timedelta(freq).total_seconds()
        interp_kwargs = {"method": "time", "limit": int(max_gap_s // freq_s)}
    else:
        interp_kwargs = {"method": "time"}

    linear_cols = ["lat", "lon", "altitude", "altitude_geom", "ground_speed", "vertical_rate"]
    discrete_cols = ["flags", "source"]

    track_rad = np.deg2rad(df["track_degrees"])
    df = df[linear_cols + discrete_cols].copy()
    df["sin_track"] = np.sin(track_rad)
    df["cos_track"] = np.cos(track_rad)

    df_combined = df.reindex(combined_idx).sort_index()

    interp_cols = linear_cols + ["sin_track", "cos_track"]
    df_combined[interp_cols] = df_combined[interp_cols].interpolate(**interp_kwargs)
    df_combined[discrete_cols] = df_combined[discrete_cols].ffill()

    out = df_combined.reindex(target_idx)

    out["track_degrees"] = (np.rad2deg(np.arctan2(out["sin_track"], out["cos_track"])) + 360) % 360
    out = out.drop(columns=["sin_track", "cos_track"])

    out["icao"] = icao
    out["flight_id"] = flight_id
    out["type"] = type_
    if t0 is not None:
        out["t0"] = t0

    return out.reset_index().rename(columns={"index": "timestamp"})

def resample_all(df, freq="10s", max_gap_s=None, n_jobs=-1):
    """Parallel resampling. max_gap_s=None means full interpolation across all gaps."""
    
    flights = [g for _, g in df.groupby("flight_id", sort=True)]
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(resample_flight)(f, freq, max_gap_s) 
        for f in tqdm(flights, desc="Resampling")
    )
    
    results = [r for r in results if r is not None]
    return pd.concat(results, ignore_index=True)
