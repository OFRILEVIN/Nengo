
# ============================================================
# MODEL 2 - MLP WITH LIF NEURONS
# Geophone Binary Classification
# Nengo + NengoDL
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import nengo
import nengo_dl

from scipy.special import expit

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

INPUT_DIM = 2000
HIDDEN_DIM = 64
OUTPUT_DIM = 1

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30

# Early stopping
PATIENCE = 5
MIN_DELTA = 1e-4

# LIF simulation
DT = 0.001
N_STEPS = 20

# Output filtering
SYNAPSE = 0.02

# LIF neuron amplitude
LIF_AMPLITUDE = 0.01

PARAMS_PATH = "model2_lif_mlp_params"

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("segmented_geophone_data.csv")

# Keep only binary labels
df_binary = df[
    df["label"].isin([0, 1])
].copy()

X = df_binary.drop(
    columns=["label"]
).to_numpy(dtype=np.float32)

y = df_binary["label"].to_numpy(
    dtype=np.int32
)

assert X.shape[1] == INPUT_DIM

print("\n===== DATASET =====")

print("Total samples:", len(X))
print("Input dimension:", X.shape[1])

print(
    "Class distribution:",
    np.bincount(y, minlength=2)
)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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

print("\n===== DATA SPLIT =====")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print(
    "Training classes:",
    np.bincount(y_train, minlength=2)
)

print(
    "Validation classes:",
    np.bincount(y_val, minlength=2)
)

print(
    "Test classes:",
    np.bincount(y_test, minlength=2)
)


# ============================================================
# 4. PREPARE DATA FOR NENGODL
# ============================================================

# Hold every complete signal constant for N_STEPS.
#
# Shape:
# (samples, simulation_steps, input_features)

def expand_inputs(X_data):

    return np.broadcast_to(
        X_data[:, None, :],
        (len(X_data), N_STEPS, INPUT_DIM)
    ).copy()


def expand_labels(y_data):

    return np.broadcast_to(
        y_data[:, None, None],
        (len(y_data), N_STEPS, OUTPUT_DIM)
    ).astype(np.float32).copy()


X_train_ng = expand_inputs(X_train)
X_val_ng = expand_inputs(X_val)
X_test_ng = expand_inputs(X_test)

Y_train_ng = expand_labels(y_train)
Y_val_ng = expand_labels(y_val)

print("\n===== NENGODL DATA SHAPES =====")

print("X_train:", X_train_ng.shape)
print("Y_train:", Y_train_ng.shape)
print("X_val:", X_val_ng.shape)
print("Y_val:", Y_val_ng.shape)
print("X_test:", X_test_ng.shape)


# ============================================================
# 5. PADDING TO FULL MINIBATCHES
# ============================================================

def pad_to_batch(X_data, Y_data=None):

    n = len(X_data)

    extra = (-n) % BATCH_SIZE

    if extra == 0:
        return X_data, Y_data

    indices = np.arange(extra) % n

    X_padded = np.concatenate(
        [X_data, X_data[indices]],
        axis=0
    )

    if Y_data is None:
        return X_padded, None

    Y_padded = np.concatenate(
        [Y_data, Y_data[indices]],
        axis=0
    )

    return X_padded, Y_padded


X_train_pad, Y_train_pad = pad_to_batch(
    X_train_ng,
    Y_train_ng
)

X_val_pad, Y_val_pad = pad_to_batch(
    X_val_ng,
    Y_val_ng
)

X_test_pad, _ = pad_to_batch(
    X_test_ng
)

print("\n===== PADDED DATA =====")

print("Training:", X_train_pad.shape)
print("Validation:", X_val_pad.shape)
print("Testing:", X_test_pad.shape)


# ============================================================
# 6. BUILD LIF MLP
# ============================================================

# Architecture:
#
# Input (2000)
#       |
# Dense (64)
#       |
# LIF (64 neurons)
#       |
# Dense (1)
#
# No recurrent connections.
# No feedback.
# No RNN.

with nengo.Network(seed=SEED) as net:

    nengo_dl.configure_settings(
        stateful=False,
        use_loop=True,
        lif_smoothing=0.01
    )

    # Input node
    inp = nengo.Node(
        np.zeros(INPUT_DIM),
        label="geophone_input"
    )

    # Hidden Dense layer
    hidden = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=HIDDEN_DIM,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=SEED
            ),
            bias_initializer="zeros"
        )
    )(inp)

    # LIF activation
    hidden = nengo_dl.Layer(
        nengo.LIF(
            amplitude=LIF_AMPLITUDE
        )
    )(hidden)

    # Output Dense layer
    output = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=OUTPUT_DIM,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=SEED + 1
            ),
            bias_initializer="zeros"
        )
    )(hidden)

    # Filtered output probe.
    # The same probe is used for training and inference.
    output_probe = nengo.Probe(
        output,
        synapse=SYNAPSE,
        label="filtered_classification_output"
    )


# Number of trainable parameters

n_parameters = (
    INPUT_DIM * HIDDEN_DIM
    + HIDDEN_DIM
    + HIDDEN_DIM * OUTPUT_DIM
    + OUTPUT_DIM
)

print("\n===== MODEL ARCHITECTURE =====")

print("Input:", INPUT_DIM)
print("Hidden LIF neurons:", HIDDEN_DIM)
print("Output:", OUTPUT_DIM)

print("Trainable parameters:", n_parameters)


# ============================================================
# 7. EARLY STOPPING
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
# 8. TRAIN THE NETWORK
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
            inp: X_train_pad
        },

        y={
            output_probe: Y_train_pad
        },

        validation_data=(

            {
                inp: X_val_pad
            },

            {
                output_probe: Y_val_pad
            }
        ),

        epochs=EPOCHS,

        shuffle=True,

        callbacks=[
            early_stopping
        ],

        verbose=2
    )

    # Save best restored parameters
    sim.save_params(PARAMS_PATH)


# ============================================================
# 9. BEST MODEL INFORMATION
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
# 10. TRAINING CURVES
# ============================================================

epochs_axis = np.arange(
    1,
    len(train_losses) + 1
)

plt.figure(figsize=(8, 5))

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
    "MLP with LIF - Training Performance"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model2_lif_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. SPIKING INFERENCE
# ============================================================

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

        outputs = sim.predict(
            {
                inp: batch
            }
        )

        filtered_output = outputs[
            output_probe
        ]

        # Expected:
        # (batch_size, N_STEPS, 1)

        if filtered_output.ndim != 3:

            raise ValueError(
                "Unexpected output shape: "
                + str(filtered_output.shape)
            )

        if filtered_output.shape[1] != N_STEPS:

            raise ValueError(
                f"Expected {N_STEPS} time steps, "
                f"received {filtered_output.shape[1]}"
            )

        if not np.all(
            np.isfinite(filtered_output)
        ):

            raise ValueError(
                "NaN or Inf detected in simulation output."
            )

        # Average filtered output across time.
        # Same temporal input is used during training.

        batch_logits = filtered_output[
            :, :, 0
        ].mean(axis=1)

        all_logits.append(
            batch_logits
        )

    return np.concatenate(
        all_logits
    )


# ============================================================
# 12. RUN SPIKING VALIDATION AND TEST
# ============================================================

with nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE,
    dt=DT
) as sim:

    # Load best model
    sim.load_params(PARAMS_PATH)

    print("\n===== SPIKING VALIDATION =====")

    val_logits_all = predict_spiking(
        sim,
        X_val_pad
    )

    val_logits = val_logits_all[
        :len(y_val)
    ]

    print("Validation output shape:", val_logits.shape)

    print("\n===== SPIKING TEST =====")

    test_logits_all = predict_spiking(
        sim,
        X_test_pad
    )

    test_logits = test_logits_all[
        :len(y_test)
    ]

    print("Test output shape:", test_logits.shape)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

# Fixed threshold:
#
# Sigmoid(logit) >= 0.5 -> Class 1
# Sigmoid(logit) < 0.5  -> Class 0
#
# Equivalent to Logit >= 0.

def evaluate_model(
    y_true,
    logits,
    name
):

    probabilities = expit(logits)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        logits
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0
    )

    print(f"\n===== {name} =====")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

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

    results = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC-AUC": auc
    }

    return (
        results,
        cm,
        probabilities,
        predictions
    )


# ============================================================
# 14. VALIDATION EVALUATION
# ============================================================

val_results, val_cm, val_probs, val_preds = (
    evaluate_model(
        y_val,
        val_logits,
        "VALIDATION - LIF MLP"
    )
)


# ============================================================
# 15. FINAL TEST EVALUATION
# ============================================================

test_results, test_cm, test_probs, test_preds = (
    evaluate_model(
        y_test,
        test_logits,
        "TEST - LIF MLP"
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=[0, 1]
)

disp.plot(cmap="Blues")

plt.title(
    "LIF MLP - Test Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "model2_lif_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 17. ROC CURVE
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
    label=f"ROC Curve (AUC = {auc:.4f})"
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
    "ROC Curve - LIF MLP"
)

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model2_lif_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. VALIDATION LOGIT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

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

plt.xlabel("Spiking Logit")
plt.ylabel("Number of Signals")

plt.title(
    "LIF MLP - Validation Logit Distribution"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "model2_lif_logits.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame([

    {
        "Dataset": "Validation",
        **val_results
    },

    {
        "Dataset": "Test",
        **test_results
    }

])

results_df.to_csv(
    "model2_lif_results.csv",
    index=False
)


predictions_df = pd.DataFrame({

    "true_label": y_test,

    "logit": test_logits,

    "probability": test_probs,

    "prediction": test_preds

})

predictions_df.to_csv(
    "model2_lif_test_predictions.csv",
    index=False
)


np.savez(
    "model2_lif_configuration.npz",

    best_epoch=best_epoch,

    best_val_loss=best_val_loss,

    threshold=0.5,

    input_dim=INPUT_DIM,

    hidden_dim=HIDDEN_DIM,

    learning_rate=LEARNING_RATE,

    batch_size=BATCH_SIZE,

    n_steps=N_STEPS,

    dt=DT,

    synapse=SYNAPSE,

    patience=PATIENCE
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n===================================")
print("       MODEL 2 - FINAL SUMMARY")
print("===================================")

print("Architecture: 2000 -> 64 LIF -> 1")

print("Parameters:", n_parameters)

print("Best epoch:", best_epoch)

print("Best validation loss:", best_val_loss)

print("Epochs trained:", len(train_losses))

print("Simulation duration (ms):", N_STEPS * DT * 1000)

print("Decision threshold: 0.5")

print("\nTest ROC-AUC:", test_results["ROC-AUC"])

print("Test Accuracy:", test_results["Accuracy"])

print(
    "Test Balanced Accuracy:",
    test_results["Balanced Accuracy"]
)

print("\nAll results saved successfully.")

print("Experiment completed.")
