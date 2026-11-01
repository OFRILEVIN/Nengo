
# ============================================================
# SECTION 1 - IMPORTS AND REPRODUCIBILITY
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nengo
import nengo_dl
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# SECTION 2 - LOAD AND FILTER DATASET
# ============================================================

df = pd.read_csv("segmented_geophone_data.csv")

# Keep only binary classes 0 and 1
df_binary = df[df["label"].isin([0, 1])].copy()

# Separate signals and labels
X = df_binary.drop(columns=["label"]).to_numpy(
    dtype=np.float32
)

y = df_binary["label"].to_numpy(dtype=np.int32)

print("Dataset shape:", X.shape)
print("Number of class 0:", np.sum(y == 0))
print("Number of class 1:", np.sum(y == 1))


# ============================================================
# SECTION 3 - TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 70% Training, 15% Validation, 15% Testing

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

# Keep original test labels for final evaluation
y_test_original = y_test.copy()

print("\nOriginal data splits:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# SECTION 4 - MINI-BATCH PADDING
# ============================================================

BATCH_SIZE = 16

def pad_to_batch_size(X_data, y_data, batch_size, seed=42):
    """
    Add randomly selected samples from the same dataset
    until the number of samples is divisible by batch_size.

    This prevents NengoDL from discarding incomplete batches.
    """

    n_extra = (-len(X_data)) % batch_size

    if n_extra == 0:
        return X_data, y_data

    rng = np.random.default_rng(seed)

    # Select additional samples from the existing split
    indices = rng.choice(
        len(X_data),
        size=n_extra,
        replace=True
    )

    X_extra = X_data[indices]
    y_extra = y_data[indices]

    X_padded = np.concatenate(
        [X_data, X_extra],
        axis=0
    )

    y_padded = np.concatenate(
        [y_data, y_extra],
        axis=0
    )

    return X_padded, y_padded


# Pad each dataset independently
X_train, y_train = pad_to_batch_size(
    X_train, y_train, BATCH_SIZE, SEED
)

X_val, y_val = pad_to_batch_size(
    X_val, y_val, BATCH_SIZE, SEED + 1
)

X_test, y_test = pad_to_batch_size(
    X_test, y_test, BATCH_SIZE, SEED + 2
)

print("\nAfter padding:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# SECTION 5 - RESHAPE DATA FOR NENGODL
# ============================================================

# NengoDL input format:
# (number of signals, simulation steps, input features)

# Each complete signal is processed in one simulation step.

X_train = X_train[:, None, :]
X_val = X_val[:, None, :]
X_test = X_test[:, None, :]

# Target format:
# (number of signals, simulation steps, output features)

y_train = y_train[:, None, None].astype(np.float32)
y_val = y_val[:, None, None].astype(np.float32)

print("\nNengoDL input shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# SECTION 6 - BUILD MLP ARCHITECTURE
# ============================================================

INPUT_DIM = X_train.shape[-1]
HIDDEN_DIM = 64
OUTPUT_DIM = 1

with nengo.Network(seed=SEED) as net:

    # Configure NengoDL
    nengo_dl.configure_settings(
        stateful=False,
        use_loop=False
    )

    # Input layer
    inp = nengo.Node(
        np.zeros(INPUT_DIM),
        label="input"
    )

    # Hidden fully connected layer
    hidden = nengo_dl.Layer(
        tf.keras.layers.Dense(
            HIDDEN_DIM,
            activation="relu"
        )
    )(inp)

    # Output layer: one raw logit
    output = nengo_dl.Layer(
        tf.keras.layers.Dense(OUTPUT_DIM)
    )(hidden)

    # Record network output
    output_probe = nengo.Probe(output)


# ============================================================
# SECTION 7 - COUNT TRAINABLE PARAMETERS
# ============================================================

# Hidden layer: weights + biases
params_hidden = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM

# Output layer: weights + biases
params_output = HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM

total_params = params_hidden + params_output

print("\n===== MODEL ARCHITECTURE =====")
print(f"Input neurons: {INPUT_DIM}")
print(f"Hidden neurons: {HIDDEN_DIM}")
print(f"Output neurons: {OUTPUT_DIM}")

print(f"Hidden layer parameters: {params_hidden:,}")
print(f"Output layer parameters: {params_output:,}")
print(f"Total trainable parameters: {total_params:,}")


# ============================================================
# SECTION 8 - CONFIGURE NENGODL SIMULATOR
# ============================================================

LEARNING_RATE = 0.001
MAX_EPOCHS = 100

sim = nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE
)

# Binary Cross-Entropy operating directly on logits
loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

sim.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss={
        output_probe: loss_fn
    },

    metrics={
        output_probe: tf.keras.metrics.BinaryAccuracy(
            threshold=0.0
        )
    }
)

print("\nSimulator configured successfully.")


# ============================================================
# SECTION 9 - EARLY STOPPING CALLBACK
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    min_delta=0.001,

    patience=5,

    mode="min",

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# SECTION 10 - TRAIN THE NETWORK
# ============================================================

history = sim.fit(

    x={inp: X_train},

    y={output_probe: y_train},

    validation_data=(
        {inp: X_val},
        {output_probe: y_val}
    ),

    epochs=MAX_EPOCHS,

    callbacks=[early_stopping],

    verbose=1,

    shuffle=True
)


# ============================================================
# SECTION 11 - IDENTIFY AND SAVE BEST MODEL
# ============================================================

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

train_accuracy = history.history["probe_binary_accuracy"]
val_accuracy = history.history["val_probe_binary_accuracy"]

# Find epoch with minimum validation loss
best_epoch = int(np.argmin(val_loss)) + 1

best_val_loss = float(np.min(val_loss))

print("\n===== EARLY STOPPING RESULTS =====")

print("Maximum epochs:", MAX_EPOCHS)
print("Epochs completed:", len(train_loss))
print("Best epoch:", best_epoch)

print(f"Best validation loss: {best_val_loss:.6f}")

print(f"Training loss at best epoch: {train_loss[best_epoch-1]:.6f}")

print(f"Validation accuracy at best epoch: {val_accuracy[best_epoch-1]:.4f}")

# IMPORTANT:
# EarlyStopping with restore_best_weights=True
# has restored the weights from best_epoch.
#
# Save parameters only AFTER the restoration.

sim.save_params("geophone_mlp_best_params")

print("\nBest model parameters saved.")


# ============================================================
# SECTION 12 - PLOT TRAINING PERFORMANCE
# ============================================================

epochs = np.arange(1, len(train_loss) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Loss
axes[0].plot(
    epochs,
    train_loss,
    label="Training Loss"
)

axes[0].plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

axes[0].axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

axes[0].scatter(
    best_epoch,
    best_val_loss,
    color="red",
    zorder=5
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_title("MLP Training Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)


# Accuracy
axes[1].plot(
    epochs,
    train_accuracy,
    label="Training Accuracy"
)

axes[1].plot(
    epochs,
    val_accuracy,
    label="Validation Accuracy"
)

axes[1].axvline(
    best_epoch,
    color="gray",
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("MLP Training Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "mlp_early_stopping_performance.png",
    dpi=300
)

plt.show()


# ============================================================
# SECTION 13 - TEST PREDICTIONS
# ============================================================

# IMPORTANT:
# Predictions are performed using the restored best weights.
# No further training takes place after EarlyStopping.

test_outputs = sim.predict(
    {inp: X_test}
)[output_probe]

# Remove predictions corresponding to padded samples
n_test_original = len(y_test_original)

logits = test_outputs.reshape(-1)[:n_test_original]

# Convert logits into probabilities
probabilities = tf.math.sigmoid(logits).numpy()

# Classification threshold
THRESHOLD = 0.5

predictions = (
    probabilities >= THRESHOLD
).astype(int)

# Use only original test labels
true_labels = y_test_original.astype(int)

# Verify matching dimensions
assert len(predictions) == len(true_labels)
assert len(probabilities) == len(true_labels)

print("\nTest samples:", len(true_labels))


# ============================================================
# SECTION 14 - EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

balanced_acc = balanced_accuracy_score(
    true_labels,
    predictions
)

auc = roc_auc_score(
    true_labels,
    probabilities
)

print("\n===== FINAL TEST RESULTS =====")

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SECTION 15 - CONFUSION MATRIX
# ============================================================

# The confusion matrix is calculated using:
# 1. Original test labels.
# 2. Predictions from the restored best model.
# 3. No padded samples.

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"MLP Test Confusion Matrix (Best Epoch: {best_epoch})"
)

plt.tight_layout()

plt.savefig(
    "mlp_best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================================================
# SECTION 16 - ROC CURVE AND SAVE RESULTS
# ============================================================

from sklearn.metrics import roc_curve

# ------------------------------------------------------------
# 1. ROC CURVE
# ------------------------------------------------------------

# Calculate ROC curve using test probabilities
# probabilities are obtained from the restored best model

fpr, tpr, thresholds = roc_curve(
    true_labels,
    probabilities
)

# AUC was already calculated in Section 14
# Reuse the existing value

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"MLP (AUC = {auc:.4f})"
)

# Random classifier reference
plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1.5,
    label="Random Classifier"
)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.title(
    f"MLP - ROC Curve (Test Set, Best Epoch: {best_epoch})"
)

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "mlp_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 2. SAVE TEST PROBABILITIES AND LABELS
# ------------------------------------------------------------

# Save predictions for future comparison with other models
# Only original test samples are included (no padding)

np.save(
    "mlp_test_probabilities.npy",
    probabilities
)

np.save(
    "mlp_test_labels.npy",
    true_labels
)

print("\nROC-AUC:", f"{auc:.4f}")

print("\nROC curve and test predictions saved successfully.")


# ============================================================
# SECTION 17 - RELEASE RESOURCES
# ============================================================

sim.close()

print("\nTraining and evaluation completed successfully.")
