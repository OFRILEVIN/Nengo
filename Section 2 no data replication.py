
# ============================================================
# GEOPHONE - THREE LIF MODELS ON CORRECTED DATA
#
# Model 1: MLP + LIF
# Model 2: STFT + 2D CNN + LIF
# Model 3: Raw 1D CNN + LIF
#
# Input: first 1000 samples of each original signal
# Training: NengoDL
# Evaluation: Spiking LIF simulation
# ============================================================

import random
from pathlib import Path

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
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

DATA_PATH = "segmented_geophone_data.csv"
OUT_DIR = Path("results_lif_nonoverlap_1000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000
KEEP_SAMPLES = 1000

# STFT configuration
NPERSEG = 128
NOVERLAP = 96

# Training
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50

# Early Stopping
PATIENCE = 5
MIN_DELTA = 1e-4

# LIF simulation
N_STEPS = 20
DT = 0.001
SYNAPSE = 0.02
LIF_AMPLITUDE = 0.01
LIF_SMOOTHING = 0.01

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD AND FILTER DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Keep classes 0 and 1
df_binary = df[
    df["label"].isin([0, 1])
].copy()

X_full = df_binary.drop(
    columns=["label"]
).to_numpy(dtype=np.float32)

y = df_binary["label"].to_numpy(dtype=np.int32)

assert X_full.shape[1] == 2000

# Remove overlapping second half
X = X_full[:, :KEEP_SAMPLES].copy()

del X_full

print("\n===== DATASET =====")
print("X shape:", X.shape)
print("Class distribution:", np.bincount(y))
print("Signal duration:", KEEP_SAMPLES / FS, "seconds")


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Same split as the ReLU experiment

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

print("\n===== SPLIT =====")
print("Train:", X_train.shape, np.bincount(y_train))
print("Validation:", X_val.shape, np.bincount(y_val))
print("Test:", X_test.shape, np.bincount(y_test))


# ============================================================
# 4. COMPUTE STFT
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

assert (N_FREQ, N_TIME) == (65, 28)

# Calculate preprocessing scale from TRAIN only
STFT_SCALE = max(
    float(np.percentile(S_train, 90)),
    1e-8
)

S_train = np.log1p(
    S_train / STFT_SCALE
).astype(np.float32)

S_val = np.log1p(
    S_val / STFT_SCALE
).astype(np.float32)

S_test = np.log1p(
    S_test / STFT_SCALE
).astype(np.float32)

print("\n===== STFT =====")
print("STFT shape:", (N_FREQ, N_TIME))
print("STFT scale:", STFT_SCALE)


# ============================================================
# 5. PREPARE MODEL FEATURES
# ============================================================

def flatten_features(data):

    return np.ascontiguousarray(
        data.reshape(len(data), -1),
        dtype=np.float32
    )


features = {
    "MLP_LIF": (
        flatten_features(X_train),
        flatten_features(X_val),
        flatten_features(X_test)
    ),

    "STFT_CNN_LIF": (
        flatten_features(S_train),
        flatten_features(S_val),
        flatten_features(S_test)
    ),

    "Raw_1D_CNN_LIF": (
        flatten_features(X_train),
        flatten_features(X_val),
        flatten_features(X_test)
    )
}


# ============================================================
# 6. MINIBATCH PADDING
# ============================================================

def pad_to_batch(X_data, y_data=None):

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


# ============================================================
# 7. EXPAND INPUTS OVER SIMULATION TIME
# ============================================================

def inputs_over_time(X_data):

    X_data = np.asarray(
        X_data,
        dtype=np.float32
    )

    return np.broadcast_to(
        X_data[:, None, :],
        (
            len(X_data),
            N_STEPS,
            X_data.shape[1]
        )
    )


def labels_over_time(y_data):

    y_data = np.asarray(
        y_data,
        dtype=np.float32
    )

    return np.broadcast_to(
        y_data[:, None, None],
        (
            len(y_data),
            N_STEPS,
            1
        )
    )


# ============================================================
# 8. BUILD LIF MODELS
# ============================================================

def build_model(model_name):

    with nengo.Network(seed=SEED, label=model_name) as net:

        nengo_dl.configure_settings(
            stateful=False,
            use_loop=True,
            lif_smoothing=LIF_SMOOTHING
        )

        # ----------------------------------------------------
        # MODEL 1: MLP + LIF
        # ----------------------------------------------------

        if model_name == "MLP_LIF":

            inp = nengo.Node(
                np.zeros(KEEP_SAMPLES),
                label="raw_input"
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    units=64,
                    activation=None
                )
            )(inp)

            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            output = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None
                )
            )(x)

            n_parameters = (
                (KEEP_SAMPLES * 64 + 64)
                + (64 + 1)
            )

        # ----------------------------------------------------
        # MODEL 2: STFT + 2D CNN + LIF
        # ----------------------------------------------------

        elif model_name == "STFT_CNN_LIF":

            INPUT_DIM = N_FREQ * N_TIME

            inp = nengo.Node(
                np.zeros(INPUT_DIM),
                label="stft_input"
            )

            CONV1_H = (N_FREQ + 1) // 2
            CONV1_W = (N_TIME + 1) // 2

            CONV2_H = (CONV1_H + 1) // 2
            CONV2_W = (CONV1_W + 1) // 2

            # Conv2D 1
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

            # LIF 1
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Conv2D 2
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

            # LIF 2
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Output
            output = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None
                )
            )(x)

            n_parameters = (
                (3 * 3 * 1 * 8 + 8)
                + (3 * 3 * 8 * 16 + 16)
                + (CONV2_H * CONV2_W * 16 + 1)
            )

        # ----------------------------------------------------
        # MODEL 3: RAW 1D CNN + LIF
        # ----------------------------------------------------

        elif model_name == "Raw_1D_CNN_LIF":

            inp = nengo.Node(
                np.zeros(KEEP_SAMPLES),
                label="raw_input"
            )

            # Input length: 1000
            # Conv1 output length: 250
            # Conv2 output length: 50
            # Conv3 output length: 13

            # Conv1D 1
            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    filters=16,
                    kernel_size=15,
                    strides=4,
                    padding="same",
                    activation=None
                )
            )(
                inp,
                shape_in=(KEEP_SAMPLES, 1)
            )

            # LIF 1
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Conv1D 2
            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=9,
                    strides=5,
                    padding="same",
                    activation=None
                )
            )(
                x,
                shape_in=(250, 16)
            )

            # LIF 2
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Conv1D 3
            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=7,
                    strides=4,
                    padding="same",
                    activation=None
                )
            )(
                x,
                shape_in=(50, 32)
            )

            # LIF 3
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Dense 32
            # NengoDL passes flattened layer output
            x = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    units=32,
                    activation=None
                )
            )(x)

            # LIF 4
            x = nengo_dl.Layer(
                nengo.LIF(
                    amplitude=LIF_AMPLITUDE
                )
            )(x)

            # Output
            output = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    units=1,
                    activation=None
                )
            )(x)

            n_parameters = (
                (15 * 1 + 1) * 16
                + (9 * 16 + 1) * 32
                + (7 * 32 + 1) * 32
                + (13 * 32 + 1) * 32
                + (32 + 1)
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        # Filtered output probe
        output_probe = nengo.Probe(
            output,
            synapse=SYNAPSE,
            label="classification_output"
        )

    return net, inp, output_probe, n_parameters


# ============================================================
# 9. SPIKING INFERENCE
# ============================================================

def predict_spiking(sim, inp, output_probe, X_data):

    X_padded, _ = pad_to_batch(X_data)

    all_logits = []

    for start in range(
        0,
        len(X_padded),
        BATCH_SIZE
    ):

        batch = X_padded[
            start:start + BATCH_SIZE
        ]

        outputs = sim.predict(
            {inp: inputs_over_time(batch)},
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
                f"Unexpected shape: {filtered_output.shape}. "
                f"Expected: {expected_shape}"
            )

        if not np.isfinite(filtered_output).all():
            raise ValueError(
                "NaN or Inf detected in output"
            )

        # Average filtered output over simulation time
        batch_logits = filtered_output[
            :, :, 0
        ].mean(axis=1)

        all_logits.append(batch_logits)

    logits = np.concatenate(all_logits)

    # Remove padded samples
    return logits[:len(X_data)]


# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    logits,
    model_name,
    dataset_name
):

    probabilities = expit(logits)

    # Fixed threshold
    # Logit >= 0 corresponds to probability >= 0.5
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
        "Model": model_name,
        "Dataset": dataset_name,

        "ROC-AUC": roc_auc_score(
            y_true,
            logits
        ),

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

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Specificity": specificity,

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print(f"\n===== {model_name}: {dataset_name} =====")

    for key, value in results.items():

        if key in ["Model", "Dataset"]:
            continue

        if key in ["TN", "FP", "FN", "TP"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(cm)

    return results, probabilities, predictions, cm


# ============================================================
# 11. SAVE FIGURES
# ============================================================

def save_figures(
    model_name,
    history,
    best_epoch,
    y_true,
    logits,
    cm
):

    train_losses = history.history["loss"]
    val_losses = history.history["val_loss"]

    epochs = np.arange(
        1,
        len(train_losses) + 1
    )

    # Loss curves
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
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
        f"{model_name} - Training Performance"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_loss.png",
        dpi=300
    )

    plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )

    disp.plot(cmap="Blues")

    plt.title(
        f"{model_name} - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_confusion.png",
        dpi=300
    )

    plt.close()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_true,
        logits
    )

    auc = roc_auc_score(
        y_true,
        logits
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} - Test ROC"
    )

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_roc.png",
        dpi=300
    )

    plt.close()

    # Validation logits
    # Useful for checking threshold behavior
    # Not included here because only test logits
    # are passed to this plotting function.


# ============================================================
# 12. EXPERIMENT SETTINGS FOR EACH MODEL
# ============================================================

experiment_settings = {
    "MLP_LIF": {
        "max_epochs": 50
    },

    "STFT_CNN_LIF": {
        "max_epochs": 50
    },

    "Raw_1D_CNN_LIF": {
        "max_epochs": 50
    }
}


# ============================================================
# 13. RUN THREE LIF MODELS
# ============================================================

all_results = []

for model_name, settings in experiment_settings.items():

    print("\n" + "=" * 65)
    print("RUNNING:", model_name)
    print("=" * 65)

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    Xtr, Xv, Xt = features[model_name]

    # Pad train and validation
    Xtr_pad, ytr_pad = pad_to_batch(
        Xtr,
        y_train
    )

    Xv_pad, yv_pad = pad_to_batch(
        Xv,
        y_val
    )

    print("Train shape:", Xtr.shape)
    print("Validation shape:", Xv.shape)
    print("Test shape:", Xt.shape)

    # Build model
    net, inp, output_probe, n_parameters = build_model(
        model_name
    )

    print("Trainable parameters:", n_parameters)

    # Parameter checks
    expected_parameters = {
        "MLP_LIF": 64129,
        "STFT_CNN_LIF": 3153,
        "Raw_1D_CNN_LIF": 25473
    }

    assert (
        n_parameters == expected_parameters[model_name]
    )

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )

    params_path = str(
        OUT_DIR / f"{model_name}_best"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

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

        print("\n===== TRAINING =====")

        history = sim.fit(
            x={
                inp: inputs_over_time(Xtr_pad)
            },

            y={
                output_probe: labels_over_time(ytr_pad)
            },

            validation_data=(
                {
                    inp: inputs_over_time(Xv_pad)
                },

                {
                    output_probe: labels_over_time(yv_pad)
                }
            ),

            epochs=settings["max_epochs"],
            shuffle=True,
            callbacks=[early_stopping],
            verbose=2
        )

        # After EarlyStopping, best weights are restored
        sim.save_params(params_path)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

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
    print("Best val loss:", best_val_loss)
    print("Epochs trained:", len(train_losses))

    # --------------------------------------------------------
    # SPIKING INFERENCE
    # --------------------------------------------------------

    with nengo_dl.Simulator(
        net,
        minibatch_size=BATCH_SIZE,
        dt=DT
    ) as sim:

        sim.load_params(params_path)

        print("\n===== VALIDATION INFERENCE =====")

        val_logits = predict_spiking(
            sim,
            inp,
            output_probe,
            Xv
        )

        print("\n===== TEST INFERENCE =====")

        test_logits = predict_spiking(
            sim,
            inp,
            output_probe,
            Xt
        )

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    val_results, val_probs, val_preds, val_cm = (
        evaluate_model(
            y_val,
            val_logits,
            model_name,
            "Validation"
        )
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_results, test_probs, test_preds, test_cm = (
        evaluate_model(
            y_test,
            test_logits,
            model_name,
            "Test"
        )
    )

    test_results["Best Epoch"] = best_epoch
    test_results["Best Val Loss"] = best_val_loss
    test_results["Trainable Parameters"] = n_parameters

    all_results.append(test_results)

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------

    save_figures(
        model_name,
        history,
        best_epoch,
        y_test,
        test_logits,
        test_cm
    )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    pd.DataFrame({
        "epoch": np.arange(
            1,
            len(train_losses) + 1
        ),

        "train_loss": train_losses,
        "val_loss": val_losses

    }).to_csv(
        OUT_DIR / f"{model_name}_history.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    pd.DataFrame({
        "true_label": y_test,
        "logit": test_logits,
        "probability": test_probs,
        "prediction": test_preds
    }).to_csv(
        OUT_DIR / f"{model_name}_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE VALIDATION LOGITS
    # --------------------------------------------------------

    pd.DataFrame({
        "true_label": y_val,
        "logit": val_logits,
        "probability": val_probs,
        "prediction": val_preds
    }).to_csv(
        OUT_DIR / f"{model_name}_validation_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE PARTIAL COMPARISON AFTER EACH MODEL
    # --------------------------------------------------------

    pd.DataFrame(all_results).to_csv(
        OUT_DIR / "three_lif_models_comparison.csv",
        index=False
    )

    print("\nModel completed:", model_name)


# ============================================================
# 14. FINAL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUT_DIR / "three_lif_models_comparison.csv",
    index=False
)

print("\n==============================================")
print("    THREE LIF MODELS - FINAL TEST RESULTS")
print("==============================================")

print(
    results_df[
        [
            "Model",
            "ROC-AUC",
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "Best Epoch",
            "Best Val Loss",
            "Trainable Parameters"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. SAVE CONFIGURATION
# ============================================================

np.savez(
    OUT_DIR / "experiment_configuration.npz",

    fs=FS,
    keep_samples=KEEP_SAMPLES,
    seed=SEED,

    stft_nperseg=NPERSEG,
    stft_noverlap=NOVERLAP,
    stft_scale=STFT_SCALE,

    n_freq=N_FREQ,
    n_time=N_TIME,

    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,

    n_steps=N_STEPS,
    dt=DT,
    synapse=SYNAPSE,

    lif_amplitude=LIF_AMPLITUDE,
    lif_smoothing=LIF_SMOOTHING,

    decision_threshold=0.5
)

print("\nSaved results in:", OUT_DIR.resolve())
print("Experiment completed.")
