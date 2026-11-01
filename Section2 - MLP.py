
# ============================================================
# GEOPHONE CLASSIFICATION - SPIKING LIF MLP
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
    RocCurveDisplay
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
INPUT_DIM = 2000

HIDDEN_1 = 128
HIDDEN_2 = 64

BATCH_SIZE = 3
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 0.0005

DT = 0.001
N_STEPS = 200
SYNAPSE = 0.02

LIF_AMPLITUDE = 0.01
PARAMS_PATH = "geophone_lif_mlp_best"

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("segmented_geophone_data.csv")

# Keep only labels 0 and 1
df = df[df["label"].isin([0, 1])].copy()

X = df.drop(columns=["label"]).to_numpy(
    dtype=np.float32
)

y = df["label"].to_numpy(
    dtype=np.int32
)

assert X.shape[1] == INPUT_DIM

# 70% Train, 15% Validation, 15% Test
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

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(
        name,
        "class counts:",
        np.bincount(labels, minlength=2)
    )


# ============================================================
# 3. PREPARE NENGODL INPUT
# ============================================================

# No per-signal normalization.
# Keep amplitude information as in the original MLP.

# NengoDL expects:
# (samples, simulation_steps, features)

X_train_ng = X_train[:, None, :].astype(np.float32)
X_val_ng = X_val[:, None, :].astype(np.float32)
X_test_ng = X_test[:, None, :].astype(np.float32)

Y_train_ng = y_train[:, None, None].astype(np.float32)
Y_val_ng = y_val[:, None, None].astype(np.float32)

# Avoid silently discarding samples
assert len(X_train_ng) % BATCH_SIZE == 0
assert len(X_val_ng) % BATCH_SIZE == 0
assert len(X_test_ng) % BATCH_SIZE == 0

print("\n===== NENGODL SHAPES =====")
print("Train X:", X_train_ng.shape)
print("Train Y:", Y_train_ng.shape)
print("Validation X:", X_val_ng.shape)
print("Test X:", X_test_ng.shape)


# ============================================================
# 4. BUILD LIF MLP
# ============================================================

# Input 2000
#     |
# Dense 128
#     |
# LIF
#     |
# Dense 64
#     |
# LIF
#     |
# Dense 1 - Linear output
#
# No RNN, no feedback

with nengo.Network(seed=SEED) as net:

    # Recommended neuron initialization for NengoDL
    net.config[nengo.Ensemble].max_rates = (
        nengo.dists.Choice([100])
    )

    net.config[nengo.Ensemble].intercepts = (
        nengo.dists.Choice([0])
    )

    net.config[nengo.Connection].synapse = None

    nengo_dl.configure_settings(
        stateful=False,
        lif_smoothing=0.01
    )

    # Input node
    inp = nengo.Node(
        np.zeros(INPUT_DIM),
        label="geophone_input"
    )

    # --------------------------------------------
    # HIDDEN LAYER 1
    # --------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=HIDDEN_1,
            activation=None
        )
    )(inp)

    x = nengo_dl.Layer(
        nengo.LIF(amplitude=LIF_AMPLITUDE)
    )(x)

    # --------------------------------------------
    # HIDDEN LAYER 2
    # --------------------------------------------

    x = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=HIDDEN_2,
            activation=None
        )
    )(x)

    x = nengo_dl.Layer(
        nengo.LIF(amplitude=LIF_AMPLITUDE)
    )(x)

    # --------------------------------------------
    # OUTPUT
    # --------------------------------------------

    output = nengo_dl.Layer(
        tf.keras.layers.Dense(
            units=1,
            activation=None
        )
    )(x)

    # Training output
    output_probe = nengo.Probe(
        output,
        label="raw_output"
    )

    # Filtered output for spiking inference
    filtered_probe = nengo.Probe(
        output,
        synapse=SYNAPSE,
        label="filtered_output"
    )

n_parameters = (
    INPUT_DIM * HIDDEN_1 + HIDDEN_1
    + HIDDEN_1 * HIDDEN_2 + HIDDEN_2
    + HIDDEN_2 + 1
)

print("\n===== LIF MLP ARCHITECTURE =====")
print("Input:", INPUT_DIM)
print("Hidden 1:", HIDDEN_1, "LIF neurons")
print("Hidden 2:", HIDDEN_2, "LIF neurons")
print("Output:", 1)
print("Parameters:", n_parameters)


# ============================================================
# 5. TRAIN NETWORK
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
        }
    )

    history = sim.fit(
        x={inp: X_train_ng},
        y={output_probe: Y_train_ng},

        validation_data=(
            {inp: X_val_ng},
            {output_probe: Y_val_ng}
        ),

        epochs=MAX_EPOCHS,
        shuffle=True,

        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
        ],

        verbose=2
    )

    # Save best restored parameters
    sim.save_params(PARAMS_PATH)


# ============================================================
# 6. TRAINING CURVES
# ============================================================

train_losses = history.history["loss"]
val_losses = history.history["val_loss"]

best_epoch = int(np.argmin(val_losses) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.axvline(
    best_epoch - 1,
    color="gray",
    linestyle="--",
    label=f"Best Epoch: {best_epoch}"
)

plt.xlabel("Epoch (zero-based plot index)")
plt.ylabel("Binary Cross-Entropy")
plt.title("LIF MLP Training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("lif_mlp_loss.png", dpi=200)
plt.show()


# ============================================================
# 7. SPIKING INFERENCE
# ============================================================

# During inference, each input signal is held constant
# for 200 simulation steps = 200 ms.
#
# This is separate from the original 2-second signal.

def predict_spiking(sim, X_data):

    logits = []

    for start in range(
        0,
        len(X_data),
        BATCH_SIZE
    ):

        batch = X_data[
            start:start + BATCH_SIZE
        ]

        # Repeat input along simulation time
        batch_spiking = np.repeat(
            batch,
            N_STEPS,
            axis=1
        )

        outputs = sim.predict(
            {inp: batch_spiking},
            verbose=0
        )

        # Average filtered output over second half
        filtered = outputs[filtered_probe]

        batch_logits = filtered[
            :, N_STEPS // 2:, 0
        ].mean(axis=1)

        logits.append(batch_logits)

    return np.concatenate(logits)


with nengo_dl.Simulator(
    net,
    minibatch_size=BATCH_SIZE,
    dt=DT
) as sim:

    sim.load_params(PARAMS_PATH)

    print("\n===== SPIKING VALIDATION =====")

    val_logits = predict_spiking(
        sim,
        X_val_ng
    )

    print("\n===== SPIKING TEST =====")

    test_logits = predict_spiking(
        sim,
        X_test_ng
    )


# ============================================================
# 8. VALIDATION THRESHOLD
# ============================================================

def choose_threshold(y_true, logits):

    values = np.unique(logits)

    if len(values) == 1:
        return 0.0

    candidates = np.concatenate([
        [values[0] - 1e-5],
        (values[:-1] + values[1:]) / 2,
        [values[-1] + 1e-5]
    ])

    scores = []

    for threshold in candidates:

        pred = (
            logits >= threshold
        ).astype(int)

        scores.append(
            f1_score(
                y_true,
                pred,
                zero_division=0
            )
        )

    return float(
        candidates[np.argmax(scores)]
    )


best_threshold = choose_threshold(
    y_val,
    val_logits
)

print("\nBest validation threshold:", best_threshold)


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_model(
    y_true,
    logits,
    threshold,
    name
):

    probabilities = expit(logits)

    predictions = (
        logits >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print(f"\n===== {name} =====")

    print(
        "Accuracy:",
        accuracy_score(y_true, predictions)
    )

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(
            y_true, predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_true,
            probabilities
        )
    )

    print(
        "Specificity:",
        tn / (tn + fp) if tn + fp else 0
    )

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

    print("\nLogit Statistics:")

    for label in [0, 1]:

        values = logits[y_true == label]

        print(
            f"Class {label}: "
            f"mean={values.mean():.4f}, "
            f"std={values.std():.4f}, "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}"
        )

    return cm, predictions, probabilities


# ============================================================
# 10. VALIDATION EVALUATION
# ============================================================

evaluate_model(
    y_val,
    val_logits,
    0.0,
    "Validation - Original Threshold"
)

evaluate_model(
    y_val,
    val_logits,
    best_threshold,
    "Validation - Tuned Threshold"
)


# ============================================================
# 11. FINAL TEST
# ============================================================

test_cm, test_predictions, test_probs = evaluate_model(
    y_test,
    test_logits,
    best_threshold,
    "Test - Spiking LIF"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=[0, 1]
).plot(cmap="Blues")

plt.title("LIF MLP - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "lif_mlp_confusion.png",
    dpi=200
)

plt.show()


# ============================================================
# 13. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    test_probs,
    name="LIF MLP"
)

plt.title("LIF MLP - Test ROC")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "lif_mlp_roc.png",
    dpi=200
)

plt.show()


# ============================================================
# 14. LOGIT DISTRIBUTION
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
    best_threshold,
    color="red",
    linestyle="--",
    label="Chosen Threshold"
)

plt.xlabel("Spiking Output Logit")
plt.ylabel("Number of Signals")
plt.title("LIF MLP - Validation Logits")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "lif_mlp_logits.png",
    dpi=200
)

plt.show()


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame({
    "true_label": y_test,
    "logit": test_logits,
    "probability": test_probs,
    "prediction": test_predictions
})

results_df.to_csv(
    "lif_mlp_test_results.csv",
    index=False
)

np.savez(
    "lif_mlp_config.npz",
    threshold=best_threshold,
    input_dim=INPUT_DIM,
    hidden_1=HIDDEN_1,
    hidden_2=HIDDEN_2,
    dt=DT,
    n_steps=N_STEPS,
    synapse=SYNAPSE
)

print("\n===== EXPERIMENT FINISHED =====")
print("Best epoch:", best_epoch)
print("Number of parameters:", n_parameters)
print("Results saved successfully.")
