
# ============================================================
# GEOPHONE - THREE ORIGINAL RELU MODELS, NON-OVERLAPPING HALVES
# Model A: Raw MLP 1000 -> Dense(64, ReLU) -> Dense(1)
# Model B: STFT -> Conv2D(8, ReLU) -> Conv2D(16, ReLU) -> Dense(1)
# Model C: Raw Conv1D(16/32/32, ReLU) -> Dense(32, ReLU) -> Dense(1)
# ============================================================

import random
from pathlib import Path

import nengo
import nengo_dl
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.signal import stft
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

SEED = 42
FS = 1000
KEEP_SAMPLES = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 5
STFT_NPERSEG = 128
STFT_NOVERLAP = 96
DATA_PATH = "segmented_geophone_data.csv"
OUT_DIR = Path("results_nonoverlap_1000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. LOAD, FILTER AND CUT BEFORE SPLITTING
# ============================================================

df = pd.read_csv(DATA_PATH)
df = df.loc[df["label"].isin([0, 1])].copy()

X_full = df.drop(columns="label").to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int32)

assert X_full.shape[1] == 2000

# Keep only the first 1000 samples
X = X_full[:, :KEEP_SAMPLES].copy()
del X_full

print("Data:", X.shape, "Labels:", np.bincount(y))
print("Each example now contains", KEEP_SAMPLES / FS, "second of signal")

# All three models use exactly the same split
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

print("Splits:", len(X_train), len(X_val), len(X_test))
print("Train classes:", np.bincount(y_train))
print("Val classes:", np.bincount(y_val))
print("Test classes:", np.bincount(y_test))

# ============================================================
# 2. PREPARE RAW AND STFT FEATURES
# ============================================================

def make_stft(signals):
    f, t, z = stft(
        signals,
        fs=FS,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=None,
        padded=False,
        axis=-1
    )

    return f, t, np.abs(z).astype(np.float32)


freqs, times, S_train = make_stft(X_train)
_, _, S_val = make_stft(X_val)
_, _, S_test = make_stft(X_test)

# Calculate scale using TRAINING data only
SCALE = max(float(np.percentile(S_train, 90)), 1e-8)

S_train = np.log1p(S_train / SCALE).astype(np.float32)
S_val = np.log1p(S_val / SCALE).astype(np.float32)
S_test = np.log1p(S_test / SCALE).astype(np.float32)

N_FREQ, N_TIME = S_train.shape[1:]

assert (N_FREQ, N_TIME) == (65, 28)

print("STFT shape:", (N_FREQ, N_TIME))
print("STFT scale:", SCALE)

features = {
    "MLP_ReLU": (
        X_train,
        X_val,
        X_test
    ),

    "STFT_CNN_ReLU": (
        S_train.reshape(len(S_train), -1),
        S_val.reshape(len(S_val), -1),
        S_test.reshape(len(S_test), -1)
    ),

    "Raw_1D_CNN_ReLU": (
        X_train,
        X_val,
        X_test
    ),
}

# ============================================================
# 3. MINIBATCH PADDING / ONE-STEP NENGODL INPUT
# ============================================================

def pad_to_batch(data, labels=None):
    n_extra = (-len(data)) % BATCH_SIZE

    if n_extra == 0:
        return data, labels

    # Padding only repeats samples within the same partition
    indices = np.arange(n_extra) % len(data)

    padded = np.concatenate(
        [data, data[indices]],
        axis=0
    )

    if labels is None:
        return padded, None

    return padded, np.concatenate(
        [labels, labels[indices]],
        axis=0
    )


def nengo_x(x):
    return np.asarray(
        x,
        dtype=np.float32
    )[:, None, :]


def nengo_y(y_values):
    return np.asarray(
        y_values,
        dtype=np.float32
    )[:, None, None]


def predict_logits(sim, inp, probe, x_original):
    x_pad, _ = pad_to_batch(x_original)

    outputs = sim.predict(
        {inp: nengo_x(x_pad)},
        verbose=0
    )[probe]

    logits = np.asarray(outputs).reshape(-1)[:len(x_original)]

    if not np.isfinite(logits).all():
        raise ValueError("Non-finite model outputs")

    return logits


def bce_from_logits(labels, logits):
    labels = np.asarray(
        labels,
        dtype=np.float64
    ).reshape(-1)

    logits = np.asarray(
        logits,
        dtype=np.float64
    ).reshape(-1)

    return float(
        np.mean(
            np.logaddexp(0, logits) - labels * logits
        )
    )


# ============================================================
# 4. BUILD ORIGINAL ARCHITECTURES - RELU, NOT LIF
# ============================================================

def build_model(model_name):

    tf.random.set_seed(SEED)

    with nengo.Network(seed=SEED, label=model_name) as net:

        nengo_dl.configure_settings(
            stateful=False,
            use_loop=False
        )

        # ----------------------------------------------------
        # MODEL A: MLP
        # ----------------------------------------------------

        if model_name == "MLP_ReLU":

            inp = nengo.Node(
                np.zeros(KEEP_SAMPLES),
                label="raw_input"
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    64,
                    activation="relu"
                )
            )(inp)

            output = nengo_dl.Layer(
                tf.keras.layers.Dense(1)
            )(x)

            n_parameters = (
                KEEP_SAMPLES * 64 + 64 + 64 + 1
            )

        # ----------------------------------------------------
        # MODEL B: STFT CNN 2D
        # ----------------------------------------------------

        elif model_name == "STFT_CNN_ReLU":

            inp = nengo.Node(
                np.zeros(N_FREQ * N_TIME),
                label="stft_input"
            )

            h1 = (N_FREQ + 1) // 2
            w1 = (N_TIME + 1) // 2

            h2 = (h1 + 1) // 2
            w2 = (w1 + 1) // 2

            x = nengo_dl.Layer(
                tf.keras.layers.Conv2D(
                    8,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    activation="relu"
                )
            )(
                inp,
                shape_in=(N_FREQ, N_TIME, 1)
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Conv2D(
                    16,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    activation="relu"
                )
            )(
                x,
                shape_in=(h1, w1, 8)
            )

            output = nengo_dl.Layer(
                tf.keras.layers.Dense(1)
            )(x)

            n_parameters = (
                (3 * 3 * 1 * 8 + 8)
                + (3 * 3 * 8 * 16 + 16)
                + (h2 * w2 * 16 + 1)
            )

        # ----------------------------------------------------
        # MODEL C: RAW 1D CNN
        # ----------------------------------------------------

        elif model_name == "Raw_1D_CNN_ReLU":

            inp = nengo.Node(
                np.zeros(KEEP_SAMPLES),
                label="raw_input"
            )

            # Input: 1000
            # After Conv1: 250
            # After Conv2: 50
            # After Conv3: 13

            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    16,
                    15,
                    strides=4,
                    padding="same",
                    activation="relu"
                )
            )(
                inp,
                shape_in=(KEEP_SAMPLES, 1),
                shape_out=(250, 16)
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    32,
                    9,
                    strides=5,
                    padding="same",
                    activation="relu"
                )
            )(
                x,
                shape_in=(250, 16),
                shape_out=(50, 32)
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Conv1D(
                    32,
                    7,
                    strides=4,
                    padding="same",
                    activation="relu"
                )
            )(
                x,
                shape_in=(50, 32),
                shape_out=(13, 32)
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Flatten()
            )(
                x,
                shape_in=(13, 32),
                shape_out=(416,)
            )

            x = nengo_dl.Layer(
                tf.keras.layers.Dense(
                    32,
                    activation="relu"
                )
            )(
                x,
                shape_in=(416,),
                shape_out=(32,)
            )

            output = nengo_dl.Layer(
                tf.keras.layers.Dense(1)
            )(
                x,
                shape_in=(32,),
                shape_out=(1,)
            )

            n_parameters = (
                (15 * 1 + 1) * 16
                + (9 * 16 + 1) * 32
                + (7 * 32 + 1) * 32
                + (416 * 32 + 32)
                + (32 + 1)
            )

        else:
            raise ValueError(model_name)

        probe = nengo.Probe(
            output,
            label="classification_logit"
        )

    return net, inp, probe, n_parameters


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def evaluate(labels, logits, model_name, partition):

    probabilities = expit(logits)

    # Logit threshold = 0
    # Equivalent to sigmoid probability threshold = 0.5
    predictions = (logits >= 0).astype(int)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": model_name,
        "Partition": partition,

        "ROC-AUC": roc_auc_score(
            labels,
            logits
        ),

        "Accuracy": accuracy_score(
            labels,
            predictions
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            labels,
            predictions
        ),

        "Precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),

        "Specificity": (
            tn / (tn + fp) if (tn + fp) else 0
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print(f"\n{model_name} / {partition}:")
    print(pd.Series(result).to_string())

    print(
        classification_report(
            labels,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return result, probabilities, predictions, cm


# ============================================================
# 6. SAVE FIGURES
# ============================================================

def save_figures(
    model_name,
    train_loss,
    val_loss,
    best_epoch,
    y_true,
    logits,
    cm
):

    axis = np.arange(
        1,
        len(train_loss) + 1
    )

    # Training / Validation loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        axis,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        axis,
        val_loss,
        label="Validation Loss"
    )

    plt.axvline(
        best_epoch,
        ls="--",
        color="gray",
        label=f"Best epoch: {best_epoch}"
    )

    plt.title(model_name + " - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE from logits")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_loss.png",
        dpi=250
    )

    plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=[0, 1]
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        model_name + " - Test confusion matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_confusion.png",
        dpi=250
    )

    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(
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
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        model_name + " - Test ROC"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{model_name}_roc.png",
        dpi=250
    )

    plt.close()


# ============================================================
# 7. RUN ALL THREE MODELS
# ============================================================

experiment_settings = {
    "MLP_ReLU": {
        "max_epochs": 30,
        "min_delta": 1e-4
    },

    "STFT_CNN_ReLU": {
        "max_epochs": 50,
        "min_delta": 1e-5
    },

    "Raw_1D_CNN_ReLU": {
        "max_epochs": 100,
        "min_delta": 0.001
    },
}

all_results = []

for model_name, settings in experiment_settings.items():

    print("\n" + "=" * 65)
    print("RUNNING:", model_name)
    print("=" * 65)

    xtr, xv, xt = features[model_name]

    # Pad training data to complete minibatches
    xtr_pad, ytr_pad = pad_to_batch(
        xtr,
        y_train
    )

    net, inp, probe, n_parameters = build_model(
        model_name
    )

    best_val_loss = np.inf
    best_epoch = 0
    no_improvement = 0

    train_loss_history = []
    val_loss_history = []

    checkpoint = str(
        OUT_DIR / f"{model_name}_best"
    )

    with nengo_dl.Simulator(
        net,
        minibatch_size=BATCH_SIZE
    ) as sim:

        sim.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),

            loss={
                probe: tf.keras.losses.BinaryCrossentropy(
                    from_logits=True
                )
            },

            metrics={
                probe: tf.keras.metrics.BinaryAccuracy(
                    threshold=0.0
                )
            }
        )

        # ----------------------------------------------------
        # TRAINING + EARLY STOPPING
        # ----------------------------------------------------

        for epoch in range(
            1,
            settings["max_epochs"] + 1
        ):

            hist = sim.fit(
                {inp: nengo_x(xtr_pad)},
                {probe: nengo_y(ytr_pad)},
                epochs=1,
                shuffle=True,
                verbose=0
            )

            train_loss = float(
                hist.history["loss"][-1]
            )

            # Validation without padded duplicates
            val_logits = predict_logits(
                sim,
                inp,
                probe,
                xv
            )

            val_loss = bce_from_logits(
                y_val,
                val_logits
            )

            train_loss_history.append(
                train_loss
            )

            val_loss_history.append(
                val_loss
            )

            print(
                f"{model_name} | "
                f"epoch {epoch:03d} | "
                f"train BCE={train_loss:.4f} | "
                f"val BCE={val_loss:.4f}"
            )

            # Save best model based on validation loss
            if val_loss < best_val_loss - settings["min_delta"]:

                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0

                sim.save_params(checkpoint)

            else:
                no_improvement += 1

            if no_improvement >= PATIENCE:
                print("Early stopping triggered")
                break

        # ----------------------------------------------------
        # RESTORE BEST MODEL
        # ----------------------------------------------------

        sim.load_params(checkpoint)

        val_logits = predict_logits(
            sim,
            inp,
            probe,
            xv
        )

        test_logits = predict_logits(
            sim,
            inp,
            probe,
            xt
        )

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    val_result, _, _, _ = evaluate(
        y_val,
        val_logits,
        model_name,
        "Validation"
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    test_result, probs, preds, cm = evaluate(
        y_test,
        test_logits,
        model_name,
        "Test"
    )

    test_result["Best Epoch"] = best_epoch
    test_result["Best Val Loss"] = best_val_loss
    test_result["Trainable Parameters"] = n_parameters

    all_results.append(test_result)

    # --------------------------------------------------------
    # SAVE PLOTS
    # --------------------------------------------------------

    save_figures(
        model_name,
        train_loss_history,
        val_loss_history,
        best_epoch,
        y_test,
        test_logits,
        cm
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    pd.DataFrame({
        "true_label": y_test,
        "logit": test_logits,
        "probability": probs,
        "prediction": preds,
    }).to_csv(
        OUT_DIR / f"{model_name}_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    pd.DataFrame({
        "epoch": np.arange(
            1,
            len(train_loss_history) + 1
        ),

        "train_loss": train_loss_history,
        "val_loss": val_loss_history,

    }).to_csv(
        OUT_DIR / f"{model_name}_history.csv",
        index=False
    )


# ============================================================
# 8. FINAL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUT_DIR / "three_relu_models_comparison.csv",
    index=False
)

np.savez(
    OUT_DIR / "experiment_configuration.npz",
    fs=FS,
    keep_samples=KEEP_SAMPLES,
    seed=SEED,
    stft_nperseg=STFT_NPERSEG,
    stft_noverlap=STFT_NOVERLAP,
    stft_scale=SCALE,
    n_freq=N_FREQ,
    n_time=N_TIME
)

print("\n===== FINAL TEST COMPARISON =====")

print(
    results_df[
        [
            "Model",
            "ROC-AUC",
            "Accuracy",
            "Balanced Accuracy",
            "F1",
            "Best Epoch",
            "Best Val Loss",
            "Trainable Parameters"
        ]
    ].to_string(index=False)
)

print("\nSaved all results in:", OUT_DIR.resolve())
