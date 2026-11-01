
# ============================================================
# 1. IMPORTS AND REPRODUCIBILITY
# ============================================================

import random
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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

SEED = 42
FS = 1000

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD AND FILTER DATASET
# ============================================================

df = pd.read_csv("segmented_geophone_data.csv")

# Keep only binary classes 0 and 1
df = df[df["label"].isin([0, 1])].copy()

# Extract raw signals and labels
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int32)

INPUT_LENGTH = X.shape[1]

print("Dataset:", X.shape)
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))
print("Signal duration:", INPUT_LENGTH / FS, "seconds")


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

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

# Preserve original test labels
y_test_original = y_test.copy()
n_test_original = len(y_test_original)

print("\nOriginal data splits:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# 4. PAD DATA TO FULL MINI-BATCHES
# ============================================================

BATCH_SIZE = 16


def pad_to_batch_size(X_data, y_data, batch_size, seed):

    n_extra = (-len(X_data)) % batch_size

    if n_extra == 0:
        return X_data, y_data

    rng = np.random.default_rng(seed)

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
# 5. RESHAPE DATA FOR NENGODL
# ============================================================

# NengoDL format:
# (number of signals, simulation steps, input features)

# One simulation step per complete signal

X_train = X_train[:, None, :]
X_val = X_val[:, None, :]
X_test = X_test[:, None, :]

y_train = y_train[:, None, None].astype(np.float32)
y_val = y_val[:, None, None].astype(np.float32)

print("\nNengoDL shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# 6. BUILD 1D CNN ARCHITECTURE
# ============================================================

with nengo.Network(seed=SEED) as net:

    nengo_dl.configure_settings(
        stateful=False,
        use_loop=False
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inp = nengo.Node(
        np.zeros(INPUT_LENGTH),
        label="raw_signal_input"
    )

    # --------------------------------------------------------
    # Conv1D Layer 1
    # Input:  (2000, 1)
    # Output: (500, 16)
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=15,
            strides=4,
            padding="same",
            activation="relu"
        )
    )(
        inp,
        shape_in=(2000, 1),
        shape_out=(500, 16)
    )

    # --------------------------------------------------------
    # Conv1D Layer 2
    # Input:  (500, 16)
    # Output: (100, 32)
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=9,
            strides=5,
            padding="same",
            activation="relu"
        )
    )(
        x,
        shape_in=(500, 16),
        shape_out=(100, 32)
    )

    # --------------------------------------------------------
    # Conv1D Layer 3
    # Input:  (100, 32)
    # Output: (25, 32)
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=7,
            strides=4,
            padding="same",
            activation="relu"
        )
    )(
        x,
        shape_in=(100, 32),
        shape_out=(25, 32)
    )

    # --------------------------------------------------------
    # Flatten
    # (25, 32) -> 800
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Flatten()
    )(
        x,
        shape_in=(25, 32),
        shape_out=(800,)
    )

    # --------------------------------------------------------
    # Dense Layer
    # 800 -> 32
    # --------------------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Dense(
            32,
            activation="relu"
        )
    )(
        x,
        shape_in=(800,),
        shape_out=(32,)
    )

    # --------------------------------------------------------
    # Output Layer
    # 32 -> 1 logit
    # --------------------------------------------------------

    output = nengo_dl.Layer(
        tf.keras.layers.Dense(1)
    )(
        x,
        shape_in=(32,),
        shape_out=(1,)
    )

    output_probe = nengo.Probe(output)


# ============================================================
# 7. COUNT TRAINABLE PARAMETERS
# ============================================================

params_conv1 = (15 * 1 + 1) * 16

params_conv2 = (9 * 16 + 1) * 32

params_conv3 = (7 * 32 + 1) * 32

params_dense = 800 * 32 + 32

params_output = 32 + 1

total_params = (
    params_conv1
    + params_conv2
    + params_conv3
    + params_dense
    + params_output
)

print("\n===== CNN ARCHITECTURE =====")

print("Input: 2000 x 1")
print("Conv1: 500 x 16")
print("Conv2: 100 x 32")
print("Conv3: 25 x 32")
print("Flatten: 800")
print("Dense: 32")
print("Output: 1")

print("\n===== TRAINABLE PARAMETERS =====")

print("Conv1:", params_conv1)
print("Conv2:", params_conv2)
print("Conv3:", params_conv3)
print("Dense:", params_dense)
print("Output:", params_output)

print("Total:", total_params)


# ============================================================
# 8. CONFIGURE NENGODL SIMULATOR
# ============================================================

LEARNING_RATE = 0.001
MAX_EPOCHS = 100

sim = nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE
)

# Output is a raw logit.
# BinaryCrossentropy applies sigmoid internally.

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
# 9. EARLY STOPPING
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
# 10. TRAIN THE NETWORK
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
# 11. IDENTIFY BEST EPOCH AND SAVE PARAMETERS
# ============================================================

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

train_accuracy = history.history[
    "probe_binary_accuracy"
]

val_accuracy = history.history[
    "val_probe_binary_accuracy"
]

best_epoch = int(np.argmin(val_loss)) + 1

best_val_loss = float(np.min(val_loss))

print("\n===== EARLY STOPPING RESULTS =====")

print("Epochs completed:", len(train_loss))
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

print(
    "Training accuracy at best epoch:",
    train_accuracy[best_epoch - 1]
)

print(
    "Validation accuracy at best epoch:",
    val_accuracy[best_epoch - 1]
)

# Save parameters after restoring best weights
sim.save_params("geophone_1dcnn_best_params")


# ============================================================
# 12. VISUALIZE TRAINING PERFORMANCE
# ============================================================

epochs = np.arange(1, len(train_loss) + 1)

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

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
    linestyle="--",
    color="gray",
    label=f"Best Epoch: {best_epoch}"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].set_title("1D CNN Training Loss")
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
    linestyle="--",
    color="gray",
    label=f"Best Epoch: {best_epoch}"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("1D CNN Training Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "1dcnn_training_performance.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. TEST EVALUATION
# ============================================================

# EarlyStopping has restored the best validation-loss weights.

test_outputs = sim.predict(
    {inp: X_test}
)[output_probe]

# Remove padded test predictions
logits = test_outputs.reshape(-1)[:n_test_original]

# Convert logits to probabilities
probabilities = tf.math.sigmoid(logits).numpy()

# Classification threshold
THRESHOLD = 0.5

predictions = (
    probabilities >= THRESHOLD
).astype(int)

true_labels = y_test_original.astype(int)

assert len(predictions) == len(true_labels)

print("\nOriginal test samples:", len(true_labels))


# ============================================================
# 14. CALCULATE TEST METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

balanced_acc = balanced_accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    true_labels,
    probabilities
)

print("\n===== 1D CNN TEST RESULTS =====")

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
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
# 15. CONFUSION MATRIX
# ============================================================

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
    f"1D CNN Test Confusion Matrix "
    f"(Best Epoch: {best_epoch})"
)

plt.tight_layout()

plt.savefig(
    "1dcnn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Best Epoch",
        "Best Validation Loss",
        "Trainable Parameters"
    ],

    "Value": [
        accuracy,
        balanced_acc,
        precision,
        recall,
        f1,
        auc,
        best_epoch,
        best_val_loss,
        total_params
    ]
})

results.to_csv(
    "1dcnn_results.csv",
    index=False
)

np.save(
    "1dcnn_test_probabilities.npy",
    probabilities
)

np.save(
    "1dcnn_test_predictions.npy",
    predictions
)

print("\nResults saved successfully.")




fpr, tpr, thresholds = roc_curve(
    true_labels,
    probabilities
)

auc = roc_auc_score(
    true_labels,
    probabilities
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
plt.title("ROC Curve - 1D CNN")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("1dcnn_roc_curve.png", dpi=300)
plt.show()

print(f"ROC-AUC: {auc:.4f}")


# ============================================================
# 17. CLOSE SIMULATOR
# ============================================================

sim.close()

print("\n1D CNN training and evaluation completed.")
