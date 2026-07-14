import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def add_derived_signals(df_flight):
    """Compute derived motion signals for a single flight (sorted by timestamp)."""
    df = df_flight.sort_values("timestamp").copy()
    
    # Time delta in seconds 
    dt = df["timestamp"].diff().dt.total_seconds()
    dt = dt.clip(upper=60)  # ignore unrealistic gaps
    
    # Acceleration: Δground_speed / Δt (kts/s)
    # Source: https://en.wikipedia.org/wiki/Acceleration
    df["acceleration"] = df["ground_speed"].diff() / dt
    
    # Vertical rate acceleration: Δvertical_rate / Δt ((ft/min)/s)
    # Source: https://en.wikipedia.org/wiki/Acceleration 
    df["vertical_acceleration"] = df["vertical_rate"].diff() / dt
    
    # Turn rate: Δheading / Δt with wrap-around correction (deg/s)
    # Source: - https://skybrary.aero/articles/rate-turn 
    #         - https://www.baeldung.com/java-compute-angle-difference
    heading_diff = df["track_degrees"].diff()
    heading_diff_wrapped = (heading_diff + 180) % 360 - 180  # wrap to [-180, 180]
    df["turn_rate"] = heading_diff_wrapped / dt
    
    # Curvature: |turn_rate| / ground_speed (1/distance)
    # Source: https://pressbooks.bccampus.ca/douglasphys1107/chapter/9-1-rotation-angle-and-angular-velocity/
    df["curvature"] = np.abs(df["turn_rate"]) / df["ground_speed"].replace(0, np.nan)


    df = df.dropna(subset=["acceleration", "vertical_acceleration", "turn_rate", "curvature"])
    return df


def add_derived_signals_all(df, n_jobs=-1):
    """Apply to all flights in parallel."""    
    flights = [g for _, g in df.groupby("flight_id", sort=False)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(add_derived_signals)(f) for f in tqdm(flights, desc="Derived signals")
    )
    return pd.concat(results, ignore_index=True)
