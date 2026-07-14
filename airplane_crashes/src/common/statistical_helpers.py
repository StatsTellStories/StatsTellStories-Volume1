import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_statistical_features(X_3d, feature_cols):

    # 1. Mean
    means = np.mean(X_3d, axis=1)
    # 2. Std
    stds = np.std(X_3d, axis=1)
    # 3. Max
    maxs = np.max(X_3d, axis=1)
    # 4. Min
    mins = np.min(X_3d, axis=1)
    # 5. Range
    ptp = maxs - mins

    # 6. Skewness & 7. Kurtosis
    skews = np.zeros((X_3d.shape[0], X_3d.shape[2]))
    kurts = np.zeros((X_3d.shape[0], X_3d.shape[2]))
    for i in range(X_3d.shape[2]):
        col = X_3d[:, :, i]
        valid = np.std(col, axis=1) > 1e-10
        if valid.any():
            skews[valid, i] = skew(col[valid], axis=1)
            kurts[valid, i] = kurtosis(col[valid], axis=1)

    # 8. RMS
    rms = np.sqrt(np.mean(X_3d**2, axis=1))

    # === FFT FEATURES ===
    fft_vals = np.abs(np.fft.rfft(X_3d, axis=1))
    # 9. Dominant frequency index
    dom_freq = np.argmax(fft_vals, axis=1).astype(float)
    # 10. Spectral energy
    spectral_energy = np.sum(fft_vals**2, axis=1)

    # 11. Spectral entropy
    fft_norm = fft_vals / (np.sum(fft_vals, axis=1, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)

    # 12. Low vs High frequency energy ratio
    mid = fft_vals.shape[1] // 2
    low_energy = np.sum(fft_vals[:, :mid, :] ** 2, axis=1)
    high_energy = np.sum(fft_vals[:, mid:, :] ** 2, axis=1)
    freq_ratio = low_energy / (high_energy + 1e-10)

    # Stack all
    X_features = np.hstack([
        means, stds, maxs, mins, ptp,
        skews, kurts, rms,
        dom_freq, spectral_energy, spectral_entropy, freq_ratio
    ])

    # Column names
    columns = []
    for stat in ['mean', 'std', 'max', 'min', 'range',
                 'skew', 'kurtosis', 'rms',
                 'dom_freq', 'spectral_energy', 'spectral_entropy', 'freq_ratio']:
        for name in feature_cols:
            columns.append(f"{name}_{stat}")

    return pd.DataFrame(X_features, columns=columns)