# -*- coding: utf-8 -*-
import numpy as np

def build_arrays_model(df, id_col, target_col, feature_cols):
    df = df.copy()
    df = df.sort_values([id_col, "timestamp"])
    n_flights = df[id_col].nunique()
    flight_ids = df[id_col].unique()
    
    n_features = len(feature_cols)
    max_timesteps = df.groupby(id_col).size().max()
    
    X_3d = np.zeros((n_flights, max_timesteps, n_features))
    y = np.zeros(n_flights)
    
    for i, f_id in enumerate(flight_ids):
        flight_data = df[df[id_col] == f_id]
        # Get the label (1 for accident, 0 for normal)
        # We take the first entry since the label must be the same for one flight_id
        y[i] = flight_data[target_col].iloc[0]
        # Extract only the feature columns
        current_features = flight_data[feature_cols].values
        
        actual_length = min(len(current_features), max_timesteps)
        # Insert into 3D array (the rest remains zeros)
        X_3d[i, :actual_length, :] = current_features[:actual_length, :]
        
    print(f"Shape of X_3d: {X_3d.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Total Accidents in y: {int(np.sum(y))}")
    return X_3d, y


