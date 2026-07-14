# -*- coding: utf-8 -*-
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def add_stl_features(df, feature_cols, flight_id_col="flight_id", period=5):
    """
    Adds STL decomposition (trend, seasonal, residual) for each feature,
    grouped by flight_id so decomposition stays within each flight.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with feature columns and a flight_id column.
    feature_cols : list of str
        Columns to decompose.
    flight_id_col : str
        Column used to group rows into individual flights.
    period : int
        Seasonal period for STL (in timesteps).

    Returns
    -------
    pd.DataFrame
        Copy of df with new columns: <feature>_trend, <feature>_seasonal, <feature>_resid
    """
    df = df.copy()
    
    flight_ids_all = df[flight_id_col].unique()
    n_total = len(flight_ids_all)

    # Pre-create empty columns to avoid fragmentation warnings
    for col in feature_cols:
        df[f"{col}_trend"] = np.nan
        df[f"{col}_seasonal"] = np.nan
        df[f"{col}_resid"] = np.nan

    for idx, (f_id, group) in enumerate(df.groupby(flight_id_col)):
        if idx % 100 == 0:
            print(f"Flight {f_id} ({idx + 1}/{n_total}) - {n_total - idx - 1} remaining...")
        idx = group.index

        # STL needs at least 2 full periods
        if len(group) < 2 * period:
            continue

        for col in feature_cols:
            series = group[col].values

            # Skip if constant (STL fails on flat signals)
            if np.all(series == series[0]):
                continue

            try:
                stl = STL(series, period=period, robust=True).fit()
                df.loc[idx, f"{col}_trend"] = stl.trend
                df.loc[idx, f"{col}_seasonal"] = stl.seasonal
                df.loc[idx, f"{col}_resid"] = stl.resid
            except Exception as e:
                print(f"STL failed for flight {f_id}, feature {col}: {e}")

    return df


def plot_stl(flight_df, feature_cols, period=10, fontsize=13):
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    label = flight_df["label"].iloc[0]
    flight_id = flight_df["flight_id"].iloc[0]
    n_features = len(feature_cols)

    fig, axes = plt.subplots(n_features, 4, figsize=(18, n_features * 3), squeeze=False)
    fig.suptitle(f"STL Decomposition - Flight {flight_id} | {'Accident' if label == 1 else 'Normal'}",
                 fontsize=fontsize + 4, fontweight="bold",
                 color="red" if label == 1 else "green", y=1.02)

    for col_title, ax in zip(["Original", "Trend", "Seasonal", "Residual"], axes[0]):
        ax.set_title(col_title, fontsize=fontsize + 1)

    for row, col in enumerate(feature_cols):
        stl = STL(flight_df[col].values, period=period, robust=True).fit()

        axes[row, 0].plot(flight_df[col].values, color="blue")
        axes[row, 1].plot(stl.trend, color="orange")
        axes[row, 2].plot(stl.seasonal, color="green")
        axes[row, 3].plot(stl.resid, color="red")

        for ax in axes[row]:
            ax.set_ylabel(col, fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
