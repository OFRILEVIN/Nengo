
# ============================================================
# MODEL 4 - STFT + CNN WITH LIF NEURONS
# Geophone Binary Classification
# Nengo + NengoDL
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import nengo
import nengo_dl

from scipy.signal import stft
from scipy.special import expit

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

FS = 1000

# STFT parameters - same as original CNN
NPERSEG = 128
NOVERLAP = 96

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50

# Early stopping
PATIENCE = 5
MIN_DELTA = 1e-4

# Spiking simulation
N_STEPS = 20
DT = 0.001
SYNAPSE = 0.02
LIF_AMPLITUDE = 0.01

PARAMS_PATH = "model4_stft_lif_cnn_best"

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD AND SPLIT DATA
# ============================================================

df = pd.read_csv("segmented_geophone_data.csv")

# Keep only labels 0 and 1
df_binary = df[
    df["label"].isin([0, 1])
].copy()

X = df_binary.drop(
    columns=["label"]
).to_numpy(dtype=np.float32)

y = df_binary["label"].to_numpy(dtype=np.int32)

assert X.shape[1] == 2000

# 70% Training, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=SEED
)

print("\n===== DATASET =====")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nClass distribution:")
print("Train:", np.bincount(y_train, minlength=2))
print("Validation:", np.bincount(y_val, minlength=2))
print("Test:", np.bincount(y_test, minlength=2))


# ============================================================
# 3. COMPUTE STFT
# ============================================================

def compute_stft(signals):

    frequencies, times, Z = stft(
        signals,
        fs=FS,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        boundary=None,
        padded=False,
        axis=-1
    )

    magnitude = np.abs(Z).astype(np.float32)

    return frequencies, times, magnitude


freqs, times, S_train = compute_stft(X_train)
_, _, S_val = compute_stft(X_val)
_, _, S_test = compute_stft(X_test)

N_FREQ = len(freqs)
N_TIME = len(times)

INPUT_DIM = N_FREQ * N_TIME

print("\n===== STFT =====")
print("Train:", S_train.shape)
print("Validation:", S_val.shape)
print("Test:", S_test.shape)

assert (N_FREQ, N_TIME) == (65, 59)


# ============================================================
# 4. LOG-MAGNITUDE PREPROCESSING
# ============================================================

# Calculate scale using training set only.
# Preserve relative amplitude information across signals.

STFT_SCALE = max(
    float(np.percentile(S_train, 90)),
    1e-8
)

S_train = np.log1p(S_train / STFT_SCALE)
S_val = np.log1p(S_val / STFT_SCALE)
S_test = np.log1p(S_test / STFT_SCALE)

print("\nSTFT scale:", STFT_SCALE)


# ============================================================
# 5. PREPARE INPUT DATA
# ============================================================

# Keep only one flattened spectrogram per signal.
# Repeat it over simulation time when needed.

def flatten_spectrogram(S):

    return np.ascontiguousarray(
        S.reshape(len(S), -1),
        dtype=np.float32
    )


X_train_flat = flatten_spectrogram(S_train)
X_val_flat = flatten_spectrogram(S_val)
X_test_flat = flatten_spectrogram(S_test)


def pad_to_batch(X_data, y_data):

    n = len(X_data)

    extra = (-n) % BATCH_SIZE

    if extra == 0:
        return X_data, y_data

    indices = np.arange(extra) % n

    X_padded = np.concatenate(
        [X_data, X_data[indices]],
        axis=0
    )

    if y_data is None:
        return X_padded, None

    y_padded = np.concatenate(
        [y_data, y_data[indices]],
        axis=0
    )

    return X_padded, y_padded


X_train_pad, y_train_pad = pad_to_batch(
    X_train_flat,
    y_train
)

X_val_pad, y_val_pad = pad_to_batch(
    X_val_flat,
    y_val
)

X_test_pad, _ = pad_to_batch(
    X_test_flat,
    None
)


# ============================================================
# 6. EXPAND DATA OVER SIMULATION TIME
# ============================================================

# NengoDL expects:
# (samples, simulation_steps, input_features)
#
# The input spectrogram is held constant during
# the simulation.

def inputs_over_time(X_data):

    return np.broadcast_to(
        X_data[:, None, :],
        (
            len(X_data),
            N_STEPS,
            INPUT_DIM
        )
    )


def labels_over_time(y_data):

    y_float = y_data.astype(np.float32)

    return np.broadcast_to(
        y_float[:, None, None],
        (
            len(y_float),
            N_STEPS,
            1
        )
    )


print("\n===== NENGODL INPUT =====")

print(
    "Train input shape:",
    inputs_over_time(X_train_pad).shape
)

print(
    "Train target shape:",
    labels_over_time(y_train_pad).shape
)


# ============================================================
# 7. BUILD CNN WITH LIF NEURONS
# ============================================================

# Original CNN:
#
# STFT -> Conv2D(8, ReLU)
#      -> Conv2D(16, ReLU)
#      -> Dense(1)
#
# LIF CNN:
#
# STFT -> Conv2D(8) -> LIF
#      -> Conv2D(16) -> LIF
#      -> Dense(1)
#
# No recurrent connections or feedback.

CONV1_H = (N_FREQ + 1) // 2
CONV1_W = (N_TIME + 1) // 2

CONV2_H = (CONV1_H + 1) // 2
CONV2_W = (CONV1_W + 1) // 2


with nengo.Network(seed=SEED) as net:

    nengo_dl.configure_settings(
        stateful=False,
        use_loop=True,
        lif_smoothing=0.01
    )

    # Input node
    inp = nengo.Node(
        np.zeros(INPUT_DIM),
        label="spectrogram_input"
    )

    # --------------------------------------------------------
    # CONVOLUTIONAL LAYER 1
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            activation=None
        )
    )(
        inp,
        shape_in=(N_FREQ, N_TIME, 1)
    )

    # LIF activation
    x = nengo_dl.Layer(
        nengo.LIF(
            amplitude=LIF_AMPLITUDE
        )
    )(x)

    # --------------------------------------------------------
    # CONVOLUTIONAL LAYER 2
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            activation=None
        )
    )(
        x,
        shape_in=(CONV1_H, CONV1_W, 8)
    )

    # LIF activation
    x = nengo_dl.Layer(
        nengo.LIF(
            amplitude=LIF_AMPLITUDE
        )
    )(x)

    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------

    output = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=1,
            activation=None
        )
    )(x)

    # Filtered output
    # Same probe used during training and inference
    output_probe = nengo.Probe(
        output,
        synapse=SYNAPSE,
        label="classification_output"
    )


# ============================================================
# 8. NUMBER OF TRAINABLE PARAMETERS
# ============================================================

n_parameters = (
    (3 * 3 * 1 * 8 + 8)
    + (3 * 3 * 8 * 16 + 16)
    + (CONV2_H * CONV2_W * 16 + 1)
)

print("\n===== CNN ARCHITECTURE =====")

print("Input:", (N_FREQ, N_TIME, 1))

print(
    "Conv1:",
    (CONV1_H, CONV1_W, 8)
)

print(
    "Conv2:",
    (CONV2_H, CONV2_W, 16)
)

print("Output:", 1)
print("Trainable parameters:", n_parameters)

assert n_parameters == 5329


# ============================================================
# 9. EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 10. TRAIN THE NETWORK
# ============================================================

with nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE,
    dt=DT
) as sim:

    sim.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss={
            output_probe:
                tf.keras.losses.BinaryCrossentropy(
                    from_logits=True
                )
        },

        metrics={
            output_probe:
                tf.keras.metrics.BinaryAccuracy(
                    threshold=0.0
                )
        }
    )

    print("\n===== START TRAINING =====")

    history = sim.fit(
        x={
            inp: inputs_over_time(X_train_pad)
        },

        y={
            output_probe: labels_over_time(y_train_pad)
        },

        validation_data=(
            {
                inp: inputs_over_time(X_val_pad)
            },
            {
                output_probe: labels_over_time(y_val_pad)
            }
        ),

        epochs=MAX_EPOCHS,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2
    )

    # Save best restored weights
    sim.save_params(PARAMS_PATH)


# ============================================================
# 11. BEST MODEL INFORMATION
# ============================================================

train_losses = history.history["loss"]
val_losses = history.history["val_loss"]

best_epoch = int(
    np.argmin(val_losses) + 1
)

best_val_loss = float(
    np.min(val_losses)
)

print("\n===== BEST MODEL =====")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Epochs trained:", len(train_losses))


# ============================================================
# 12. TRAINING CURVES
# ============================================================

epochs_axis = np.arange(
    1,
    len(train_losses) + 1
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs_axis,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs_axis,
    val_losses,
    label="Validation Loss"
)

plt.axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")

plt.title(
    "STFT LIF-CNN Training Performance"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model4_lif_cnn_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. SPIKING INFERENCE FUNCTION
# ============================================================

# Use actual LIF spiking activity during prediction.
# Reset simulation states between prediction calls.

def predict_spiking(sim, X_data):

    all_logits = []

    for start in range(
        0,
        len(X_data),
        BATCH_SIZE
    ):

        batch = X_data[
            start:start + BATCH_SIZE
        ]

        batch_time = inputs_over_time(batch)

        outputs = sim.predict(
            {inp: batch_time},
            n_steps=N_STEPS,
            stateful=False,
            verbose=0
        )

        filtered_output = np.asarray(
            outputs[output_probe]
        )

        expected_shape = (
            len(batch),
            N_STEPS,
            1
        )

        if filtered_output.shape != expected_shape:

            raise ValueError(
                f"Unexpected output shape: "
                f"{filtered_output.shape}. "
                f"Expected: {expected_shape}"
            )

        if not np.all(np.isfinite(filtered_output)):

            raise ValueError(
                "NaN or Inf detected in spiking output."
            )

        # Average output across simulation time
        batch_logits = filtered_output[
            :, :, 0
        ].mean(axis=1)

        all_logits.append(batch_logits)

    return np.concatenate(all_logits)


# ============================================================
# 14. SPIKING VALIDATION AND TEST
# ============================================================

with nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE,
    dt=DT
) as sim:

    sim.load_params(PARAMS_PATH)

    print("\n===== SPIKING VALIDATION =====")

    val_logits_all = predict_spiking(
        sim,
        X_val_pad
    )

    val_logits = val_logits_all[
        :len(y_val)
    ]

    print(
        "Validation output shape:",
        val_logits.shape
    )

    print("\n===== SPIKING TEST =====")

    test_logits_all = predict_spiking(
        sim,
        X_test_pad
    )

    test_logits = test_logits_all[
        :len(y_test)
    ]

    print(
        "Test output shape:",
        test_logits.shape
    )


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

# Fixed threshold:
#
# Logit >= 0 -> Class 1
# Logit < 0  -> Class 0
#
# Equivalent to sigmoid probability >= 0.5.
#
# No threshold optimization.

def evaluate_model(
    y_true,
    logits,
    dataset_name
):

    probabilities = expit(logits)

    predictions = (
        logits >= 0
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    results = {
        "Dataset": dataset_name,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            logits
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print(f"\n===== {dataset_name} RESULTS =====")

    for key, value in results.items():

        if key == "Dataset":
            continue

        if key in ["TN", "FP", "FN", "TP"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return (
        results,
        probabilities,
        predictions,
        cm
    )


# ============================================================
# 16. VALIDATION EVALUATION
# ============================================================

val_results, val_probs, val_preds, val_cm = (
    evaluate_model(
        y_val,
        val_logits,
        "Validation"
    )
)


# ============================================================
# 17. TEST EVALUATION
# ============================================================

test_results, test_probs, test_preds, test_cm = (
    evaluate_model(
        y_test,
        test_logits,
        "Test"
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=[0, 1]
)

disp.plot(cmap="Blues")

plt.title(
    "STFT LIF-CNN - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "model4_lif_cnn_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_logits
)

auc = roc_auc_score(
    y_test,
    test_logits
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"LIF-CNN (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("STFT LIF-CNN - Test ROC")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model4_lif_cnn_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. VALIDATION LOGIT DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    val_logits[y_val == 0],
    bins=25,
    alpha=0.6,
    label="Class 0"
)

plt.hist(
    val_logits[y_val == 1],
    bins=25,
    alpha=0.6,
    label="Class 1"
)

plt.axvline(
    0,
    color="red",
    linestyle="--",
    label="Decision Threshold = 0"
)

plt.xlabel("Spiking Output Logit")
plt.ylabel("Number of Signals")

plt.title(
    "STFT LIF-CNN - Validation Logits"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model4_lif_cnn_logits.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame([
    val_results,
    test_results
])

results_df.to_csv(
    "model4_lif_cnn_results.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "true_label": y_test,
    "logit": test_logits,
    "probability": test_probs,
    "prediction": test_preds
})

predictions_df.to_csv(
    "model4_lif_cnn_test_predictions.csv",
    index=False
)

np.savez(
    "model4_lif_cnn_preprocessing.npz",

    fs=FS,
    nperseg=NPERSEG,
    noverlap=NOVERLAP,

    scale=STFT_SCALE,

    n_freq=N_FREQ,
    n_time=N_TIME,

    n_steps=N_STEPS,
    dt=DT,
    synapse=SYNAPSE,

    best_epoch=best_epoch,
    best_val_loss=best_val_loss,

    threshold=0.5
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("       MODEL 4 - FINAL SUMMARY")
print("===================================")

print(
    "Architecture: STFT -> Conv8 -> LIF -> "
    "Conv16 -> LIF -> Dense1"
)

print("Parameters:", n_parameters)

print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Epochs trained:", len(train_losses))

print(
    "Simulation duration (ms):",
    N_STEPS * DT * 1000
)

print("Decision threshold: 0.5")

print(
    "\nTest ROC-AUC:",
    test_results["ROC-AUC"]
)

print(
    "Test Accuracy:",
    test_results["Accuracy"]
)

print(
    "Test Balanced Accuracy:",
    test_results["Balanced Accuracy"]
)

print("\nExperiment completed.")
