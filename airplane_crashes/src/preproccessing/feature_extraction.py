import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt



def extract_statistical_features(X_3d, feature_cols):
    """
    Transforms (Samples, Timestamps, Features) -> (Samples, StatFeatures)
    """
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

    # 6. Skewness & 7. Kurtosis: only compute on non-constant samples
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
    # Apply Fast Fourier Transform along the time axis to convert each signal from time-domain to frequency-domain.
    fft_vals = np.abs(np.fft.rfft(X_3d, axis=1))

    # 9. Dominant frequency index
    # At which frequency does the most energy occur? (per flight, per feature)
    dom_freq = np.argmax(fft_vals, axis=1).astype(float)

    # 10. Spectral energy
    # Total power in the frequency domain - high value = strong signal
    spectral_energy = np.sum(fft_vals**2, axis=1)

    # 11. Spectral entropy
    # How spread out is the energy across frequencies?
    # Low entropy = energy concentrated in few frequencies (structured signal)
    # High entropy = energy spread across many frequencies (chaotic/noisy signal)
    fft_norm = fft_vals / (np.sum(fft_vals, axis=1, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10), axis=1)

    # 12. Low vs High frequency energy ratio
    # Split spectrum in half: low frequencies = smooth trends, high frequencies = rapid changes/noise
    # High ratio = signal dominated by slow changes (normal flight)
    # Low ratio = signal dominated by rapid changes (possible anomaly)
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



def plot_fft_features_by_stat(df_plot, feature_cols):
    """
    Plots FFT feature distributions (Accident vs Non-Accident) grouped by stats.
    4 figures (one per FFT stat), each with 11 subplots (one per feature).
    """
    fft_stats = ['dom_freq', 'spectral_energy', 'spectral_entropy', 'freq_ratio']
    
    accidents = df_plot[df_plot['label'] == 1]
    normals   = df_plot[df_plot['label'] == 0]

    for stat in fft_stats:
        n_rows = int(np.ceil(len(feature_cols) / 4))
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows), squeeze=False)

        #fig, axes = plt.subplots(3, 4, figsize=(20, 12), squeeze=False)
        fig.suptitle(f"FFT Feature: {stat} | Accident vs Normal", fontsize=15, fontweight='bold')
        axes_flat = axes.flatten()

        for i, feat in enumerate(feature_cols):
            col = f"{feat}_{stat}"
            ax = axes_flat[i]
            ax.hist(normals[col], bins=40, alpha=0.6, color='steelblue', label='Normal', density=True)
            ax.hist(accidents[col], bins=40, alpha=0.6, color='red', label='Accident', density=True)
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel(stat)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)

        # hide empty subplots (3x4=12, but only 11 features)
        for j in range(len(feature_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()
