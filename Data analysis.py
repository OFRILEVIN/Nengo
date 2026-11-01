
"""
Geophone Signal Analysis
========================
Exploratory Data Analysis (EDA) on the TRAINING set only.

Sections:
    1. Imports and configuration
    2. Data loading and splitting
    3. Overall class statistics
    4. Amplitude distribution
    5. Time-dependent variability
    6. RMS distribution
    7. Mean Power Spectral Density (PSD)
    8. Normalized PSD
    9. Representative STFT spectrograms
"""

# ============================================================
# SECTION 1 - IMPORTS AND CONFIGURATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch, stft
from sklearn.model_selection import train_test_split

SEED = 42
FS = 1000  # Sampling frequency (Hz)

DATA_PATH = "segmented_geophone_data.csv"

# Welch PSD parameters
PSD_WINDOW = 512
PSD_OVERLAP = 256

# STFT parameters
STFT_WINDOW = 128
STFT_OVERLAP = 96

EPS = 1e-12

CLASS_LABELS = (0, 1)

CLASS_COLORS = {
    0: "tab:blue",
    1: "tab:orange"
}


# ============================================================
# SECTION 2 - DATA LOADING AND SPLITTING
# ============================================================

# Load the original dataset
df = pd.read_csv(DATA_PATH)

# Keep only labels 0 and 1
df = df[df["label"].isin(CLASS_LABELS)].copy()

# Separate signals and labels
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int32)

# Stratified split:
# 70% Training, 15% Validation, 15% Testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

# Separate training signals by class
class_signals = {
    label: X_train[y_train == label]
    for label in CLASS_LABELS
}

N_SAMPLES = X_train.shape[1]
TIME = np.arange(N_SAMPLES) / FS

print("\n===== DATASET INFORMATION =====")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Signal duration:", N_SAMPLES / FS, "seconds")

for label in CLASS_LABELS:
    print(
        f"Class {label}: "
        f"{len(class_signals[label])} training signals"
    )


# ============================================================
# SECTION 3 - OVERALL CLASS STATISTICS
# ============================================================

# Calculate statistics across all samples of each class.

def compute_class_statistics(signals, label):

    values = signals.ravel()

    return {
        "Class": label,
        "Signals": len(signals),
        "Samples": values.size,
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Mean": values.mean(),
        "Std": values.std(),
        "Mean Abs Amplitude": np.mean(np.abs(values))
    }


statistics = pd.DataFrame([
    compute_class_statistics(class_signals[label], label)
    for label in CLASS_LABELS
])

print("\n===== OVERALL CLASS STATISTICS =====")
print(
    statistics.to_string(
        index=False,
        float_format="%.6f"
    )
)


# ============================================================
# SECTION 4 - AMPLITUDE DISTRIBUTION
# ============================================================

# Pool all amplitude samples from each class.
# Both classes use the same histogram bins.

amplitudes = {
    label: class_signals[label].ravel()
    for label in CLASS_LABELS
}

all_amplitudes = X_train.ravel()

amplitude_bins = np.linspace(
    all_amplitudes.min(),
    all_amplitudes.max(),
    161
)

plt.figure(figsize=(11, 6))

for label in CLASS_LABELS:

    plt.hist(
        amplitudes[label],
        bins=amplitude_bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS[label],
        label=f"Class {label}"
    )

plt.title("Amplitude Distribution by Class - Training Set")
plt.xlabel("Signal Amplitude")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 5 - RMS DISTRIBUTION
# ============================================================

# Calculate one RMS value for each signal.

rms = {
    label: np.sqrt(
        np.mean(
            class_signals[label].astype(np.float64) ** 2,
            axis=1
        )
    )
    for label in CLASS_LABELS
}

all_rms = np.concatenate([
    rms[label] for label in CLASS_LABELS
])

rms_bins = np.linspace(
    all_rms.min(),
    all_rms.max(),
    61
)

plt.figure(figsize=(10, 5))

for label in CLASS_LABELS:

    plt.hist(
        rms[label],
        bins=rms_bins,
        density=True,
        alpha=0.55,
        color=CLASS_COLORS[label],
        label=f"Class {label}"
    )

plt.title("RMS Distribution by Class - Training Set")
plt.xlabel("Signal RMS")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Numerical RMS statistics
print("\n===== RMS STATISTICS =====")

for label in CLASS_LABELS:

    values = rms[label]

    print(
        f"Class {label}: "
        f"median={np.median(values):.6f}, "
        f"mean={values.mean():.6f}, "
        f"std={values.std():.6f}"
    )


# ============================================================
# SECTION 6 - MEAN POWER SPECTRAL DENSITY (PSD)
# ============================================================

# Calculate Welch PSD for all training signals.
# Each row corresponds to one signal.

freqs, psd_train = welch(
    X_train,
    fs=FS,
    window="hann",
    nperseg=PSD_WINDOW,
    noverlap=PSD_OVERLAP,
    detrend="constant",
    scaling="density",
    axis=1
)

# Separate PSDs by class
class_psd = {
    label: psd_train[y_train == label]
    for label in CLASS_LABELS
}

# Average PSD across signals in each class.
# Averaging is performed in linear units before conversion to dB.

mean_psd = {
    label: class_psd[label].mean(axis=0)
    for label in CLASS_LABELS
}

plt.figure(figsize=(11, 5))

for label in CLASS_LABELS:

    plt.plot(
        freqs,
        10 * np.log10(
            np.maximum(mean_psd[label], EPS)
        ),
        color=CLASS_COLORS[label],
        label=f"Class {label}"
    )

plt.title("Mean Power Spectral Density by Class")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB/Hz)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 7 - NORMALIZED PSD
# ============================================================

# Calculate total spectral power for each signal.
# Normalize each PSD independently to compare spectral shapes.

total_power = np.trapz(
    psd_train,
    freqs,
    axis=1
)

normalized_psd = (
    psd_train /
    np.maximum(total_power[:, None], EPS)
)

# Calculate mean normalized PSD for each class
mean_normalized_psd = {
    label: normalized_psd[y_train == label].mean(axis=0)
    for label in CLASS_LABELS
}

plt.figure(figsize=(11, 5))

for label in CLASS_LABELS:

    plt.plot(
        freqs,
        10 * np.log10(
            np.maximum(mean_normalized_psd[label], EPS)
        ),
        color=CLASS_COLORS[label],
        label=f"Class {label}"
    )

plt.title("Normalized Spectral Shape by Class - Training only")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized PSD (dB, relative to 1/Hz)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 8 - REPRESENTATIVE STFT SPECTROGRAMS
# ============================================================

# Select one representative signal per class:
# The signal whose RMS is closest to the class median.

representative_signals = {}

for label in CLASS_LABELS:

    values = rms[label]

    idx = np.argmin(
        np.abs(values - np.median(values))
    )

    representative_signals[label] = (
        class_signals[label][idx]
    )

# Calculate STFT for both representative signals.
# Signals are stacked to avoid repeating the STFT call.

signals_to_plot = np.stack([
    representative_signals[label]
    for label in CLASS_LABELS
])

stft_freqs, stft_times, Z = stft(
    signals_to_plot,
    fs=FS,
    window="hann",
    nperseg=STFT_WINDOW,
    noverlap=STFT_OVERLAP,
    boundary=None,
    padded=False,
    axis=-1
)

# Convert STFT magnitude to dB
spectrogram_db = 20 * np.log10(
    np.maximum(np.abs(Z), EPS)
)

# Use the same color scale for both spectrograms
vmin = spectrogram_db.min()
vmax = spectrogram_db.max()

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 8),
    sharex=True,
    sharey=True
)

for i, label in enumerate(CLASS_LABELS):

    im = axes[i].pcolormesh(
        stft_times,
        stft_freqs,
        spectrogram_db[i],
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax
    )

    axes[i].set_title(
        f"Class {label} - Representative Signal"
    )
    axes[i].set_ylabel("Frequency (Hz)")

axes[-1].set_xlabel("Time (s)")

fig.colorbar(
    im,
    ax=axes,
    label="STFT Magnitude (dB)"
)

plt.tight_layout()
plt.show()


# ============================================================
# END OF ANALYSIS
# ============================================================

print("\nTraining-set analysis completed successfully.")
