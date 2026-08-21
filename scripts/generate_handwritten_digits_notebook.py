"""
generate_handwritten_digits_notebook.py
----------------------------------------
Run this script once to create the full beginner-friendly notebook:
    notebooks/handwritten_digits_tensorflow_step_by_step.ipynb

Usage:
    python scripts/generate_handwritten_digits_notebook.py
"""

import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def md_cell(uid: str, text: str) -> dict:
    """Return a Jupyter markdown cell dict."""
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": uid,
        "metadata": {},
        "source": lines,
    }


def code_cell(uid: str, text: str) -> dict:
    """Return a Jupyter code cell dict."""
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": lines,
    }


# ──────────────────────────────────────────────────────────────────────
# Cell definitions
# ──────────────────────────────────────────────────────────────────────

CELLS = []

# ── 1. Title ──────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-01", """\
# Classifying Handwritten Digits with TensorFlow and Keras

**Last Updated:** 20 Aug 2026

---

## What are we doing?

We will teach a computer to **recognise handwritten digits** (0 through 9).

This is just like teaching a child to read numbers — except we are teaching a machine!

## What is MNIST?

**MNIST** is a famous beginner dataset used all over the world. It contains:

| Split     | Images  | Description                         |
|-----------|---------|-------------------------------------|
| Training  | 60,000  | Used to **teach** the model         |
| Testing   | 10,000  | Used to **check** how well it learnt |

Each image is a **28 × 28 pixel** grayscale picture of a handwritten digit.

## What will we build?

We build a **Neural Network** — a type of model loosely inspired by the human brain.
The network looks at pixel values and learns which digit is shown.

## Steps in this notebook

1. Load and explore the dataset
2. Visualise sample images
3. Normalise (clean up) the data
4. Build the neural network
5. Train the model
6. Plot training progress
7. Evaluate performance on unseen data
8. Confusion matrix
9. Sample predictions
10. Save and reload the trained model
11. Final summary\
"""))

# ── 2. Imports ─────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-02", """\
## Cell 1 — Import Libraries

Before we do anything, we load all the tools (libraries) we need.\
"""))

CELLS.append(code_cell("hd-03", """\
# Cell 1: Import all required libraries.
# Run  %pip install tensorflow seaborn scikit-learn  if any import fails.

import numpy as np                        # Numbers and arrays
import matplotlib.pyplot as plt           # Charts and images
import seaborn as sns                     # Pretty heatmaps
import tensorflow as tf                   # Neural-network framework
from sklearn.metrics import (             # Evaluation helpers
    confusion_matrix,
    classification_report,
)

# Fix random seeds so results are the same every time you run
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("All libraries loaded successfully.")
print(f"TensorFlow version: {tf.__version__}")\
"""))

# ── 3. Load data ────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-04", """\
## Step 1 — Load the MNIST Dataset

### What is happening here?
- We ask Keras to download the MNIST data automatically.
- It gives us four things:
  - `x_train` / `y_train` — images and labels for **training**
  - `x_test`  / `y_test`  — images and labels for **testing**

### Why does this step matter?
- Without data there is nothing to learn from.
- Splitting into train/test lets us measure how well the model
  works on images it has **never seen before**.\
"""))

CELLS.append(code_cell("hd-05", """\
# Cell 2: Load the MNIST dataset from Keras.

def load_mnist_data():
    \"\"\"
    Download and return the MNIST handwritten-digit dataset.

    Returns:
        (x_train, y_train): 60,000 training images and their digit labels.
        (x_test,  y_test):  10,000 test images and their digit labels.

    Notes:
        - Each image shape is (28, 28) — grayscale pixels 0..255.
        - Each label is an integer 0..9 (the digit shown).
    \"\"\"
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist_data()

print("Dataset loaded successfully!")
print(f"Training images shape : {x_train.shape}")
print(f"Training labels shape : {y_train.shape}")
print(f"Test images shape     : {x_test.shape}")
print(f"Test labels shape     : {y_test.shape}")
print(f"Pixel value range     : {x_train.min()} to {x_train.max()}")
print(f"Digit classes         : {np.unique(y_train)}")\
"""))

CELLS.append(md_cell("hd-06", """\
### Inference — Understanding the Output

| Variable   | Shape         | Meaning                                      |
|------------|---------------|----------------------------------------------|
| `x_train`  | (60000,28,28) | 60,000 images, each 28 rows × 28 columns     |
| `y_train`  | (60000,)      | 60,000 digit labels (one per image)          |
| `x_test`   | (10000,28,28) | 10,000 images kept aside for final testing   |
| `y_test`   | (10000,)      | 10,000 digit labels for the test images      |

Pixel values range from **0** (black) to **255** (white).
There are **10 classes** — one for each digit 0 through 9.\
"""))

# ── 4. Class distribution ───────────────────────────────────────────────
CELLS.append(md_cell("hd-07", """\
## Step 2 — Explore the Class Distribution

### What is happening here?
We count how many examples of each digit exist in the training set.\
"""))

CELLS.append(code_cell("hd-08", """\
# Cell 3: Plot the class distribution (how many images per digit).

def plot_class_distribution(labels, title="Class Distribution"):
    \"\"\"
    Draw a bar chart showing how many images belong to each digit class.

    Args:
        labels (np.ndarray): Array of integer digit labels (0–9).
        title  (str):        Chart title.
    \"\"\"
    unique_classes, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 4))
    bars = plt.bar(unique_classes, counts, color="steelblue", edgecolor="black")
    plt.xticks(unique_classes)
    plt.xlabel("Digit (0–9)", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")

    # Write the count on top of each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(count),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


plot_class_distribution(y_train, title="Training Set — Images per Digit")\
"""))

CELLS.append(md_cell("hd-09", """\
### Inference — Class Distribution

- Each digit has roughly **6,000** training images — the dataset is **balanced**.
- A balanced dataset is good news! It means the model gets equal practice
  for every digit and is less likely to favour one digit over another.
- If one class had far fewer examples, the model might struggle to recognise it.\
"""))

# ── 5. Visualise sample images ──────────────────────────────────────────
CELLS.append(md_cell("hd-10", """\
## Step 3 — Visualise Sample Images

### What is happening here?
We display a small grid of images so we can see what the model will learn from.\
"""))

CELLS.append(code_cell("hd-11", """\
# Cell 4: Show a grid of sample training images.

def show_sample_images(images, labels, n_rows=2, n_cols=5):
    \"\"\"
    Display a grid of sample images with their digit labels.

    Args:
        images  (np.ndarray): Array of images, shape (N, 28, 28).
        labels  (np.ndarray): Array of integer labels (0–9).
        n_rows  (int):        Number of rows in the grid.
        n_cols  (int):        Number of columns in the grid.
    \"\"\"
    plt.figure(figsize=(12, 5))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Digit: {labels[i]}", fontsize=11)
        plt.axis("off")
    plt.suptitle("Sample Images from the MNIST Training Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


show_sample_images(x_train, y_train)\
"""))

CELLS.append(md_cell("hd-12", """\
### Inference — What We See

- Every image is a **28 × 28 greyscale picture** of a handwritten digit.
- Lighter pixels are where ink was drawn; darker pixels are the background.
- Different people write the same digit differently — some are neat, some are messy.
- The model must learn to recognise all these variations.\
"""))

# ── 6. Normalise ────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-13", """\
## Step 4 — Normalise the Pixel Values

### What is happening here?
We divide every pixel value by **255** to convert the range from 0–255 to **0.0–1.0**.

### Why does this step matter?
- Think of it like converting temperatures from Fahrenheit to Celsius —
  the *pattern* stays the same but the numbers are easier to work with.
- Neural networks learn much faster and more reliably when input values
  are **small** (close to 0 and 1) rather than large (up to 255).\
"""))

CELLS.append(code_cell("hd-14", """\
# Cell 5: Normalise pixel values to the range 0.0 – 1.0.

def normalize_images(x_train, x_test):
    \"\"\"
    Scale pixel values from the range 0–255 down to 0.0–1.0.

    Why normalise?
        Large input values make gradient descent (the learning algorithm)
        unstable. Scaling to 0–1 makes training faster and more reliable.

    Args:
        x_train (np.ndarray): Raw training images (pixel values 0–255).
        x_test  (np.ndarray): Raw test images     (pixel values 0–255).

    Returns:
        x_train_norm (np.ndarray): Normalised training images.
        x_test_norm  (np.ndarray): Normalised test images.
    \"\"\"
    x_train_norm = x_train.astype("float32") / 255.0
    x_test_norm  = x_test.astype("float32")  / 255.0
    return x_train_norm, x_test_norm


x_train, x_test = normalize_images(x_train, x_test)

print(f"Pixel range BEFORE normalisation: 0 – 255")
print(f"Pixel range AFTER  normalisation: {x_train.min():.1f} – {x_train.max():.1f}")\
"""))

CELLS.append(code_cell("hd-15", """\
# Cell 6: Side-by-side comparison — before and after normalisation.

def compare_normalization(raw_image, norm_image, label):
    \"\"\"
    Show one image before and after normalisation side by side.

    Args:
        raw_image  (np.ndarray): Original image (values 0–255).
        norm_image (np.ndarray): Normalised image (values 0.0–1.0).
        label      (int):        Digit label for the image.
    \"\"\"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax1.imshow(raw_image, cmap="gray")
    ax1.set_title(f"Before Normalisation\\nDigit: {label}\\nPixel range: 0–255", fontsize=10)
    ax1.axis("off")

    ax2.imshow(norm_image, cmap="gray")
    ax2.set_title(f"After Normalisation\\nDigit: {label}\\nPixel range: 0.0–1.0", fontsize=10)
    ax2.axis("off")

    plt.suptitle("Effect of Normalisation (Image Looks the Same!)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Show for the first training image.
# We must reload the raw image because x_train is already normalised.
(x_train_raw, y_train_raw), _ = tf.keras.datasets.mnist.load_data()
compare_normalization(x_train_raw[0], x_train[0], y_train[0])\
"""))

CELLS.append(md_cell("hd-16", """\
### Inference — Normalisation

- The two images look **identical** — normalisation only changes the number scale,
  not the visual appearance.
- However, those smaller numbers make a big difference to how fast and how well
  the neural network learns.\
"""))

# ── 7. Build model ──────────────────────────────────────────────────────
CELLS.append(md_cell("hd-17", """\
## Step 5 — Build the Neural Network

### What is a Neural Network?

Imagine a row of workers passing information along a chain.
Each worker looks at what the previous worker said, does a small calculation,
and passes the result onwards.  That is essentially what a neural network does.

### Architecture we will use

```
Input image (28×28 pixels)
        ↓
Flatten  →  784 numbers (all pixels in one long list)
        ↓
Dense 128  →  128 "thinking" neurons with ReLU activation
        ↓
Dense 128  →  128 more neurons with ReLU activation
        ↓
Dense 10   →  10 output neurons (one per digit 0–9) with Softmax
```

### Key terms

| Term      | Plain English                                           |
|-----------|---------------------------------------------------------|
| Flatten   | Unroll the 28×28 grid into one list of 784 numbers      |
| Dense     | Every neuron is connected to every neuron in the layer before it |
| ReLU      | "Pass positive numbers through; replace negatives with 0"        |
| Softmax   | Convert 10 raw scores into 10 probabilities that sum to 1        |\
"""))

CELLS.append(code_cell("hd-18", """\
# Cell 7: Build the neural network model.

def build_model():
    \"\"\"
    Build a simple fully-connected neural network for digit classification.

    Architecture:
        1. Flatten (28×28 → 784): Converts the 2-D image to a 1-D list.
        2. Dense(128, ReLU):       First hidden layer — learns basic patterns.
        3. Dense(128, ReLU):       Second hidden layer — learns complex patterns.
        4. Dense(10, Softmax):     Output layer — one probability per digit (0–9).

    Returns:
        model (tf.keras.Sequential): Uncompiled neural-network model.
    \"\"\"
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(128, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(10,  activation="softmax", name="output"),
        ],
        name="digit_classifier",
    )
    return model


model = build_model()
model.summary()\
"""))

CELLS.append(md_cell("hd-19", """\
### Inference — Model Summary

The summary tells you:
- **Layer names** and types.
- **Output shape** — how many numbers pass to the next layer.
- **Param #** — number of learnable numbers (weights) in each layer.

The model has about **118,000 parameters** to learn.
During training, all these numbers are slowly adjusted until the model
makes good predictions.\
"""))

# ── 8. Train ────────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-20", """\
## Step 6 — Compile and Train the Model

### What is happening here?
- **Compile** sets up the learning strategy (how mistakes are measured and fixed).
- **Fit** runs the actual learning loop over the training images.

### Key concepts

| Concept                           | Plain English                                                        |
|-----------------------------------|----------------------------------------------------------------------|
| Optimizer (Adam)                  | The strategy for updating weights — Adam is fast and reliable        |
| Loss (sparse_categorical_crossentropy) | How we measure how wrong a prediction is — we want this to fall  |
| Accuracy                          | The percentage of correct predictions — we want this to rise         |
| Epoch                             | One complete pass through all 60,000 training images                 |
| Validation split (10 %)           | 6,000 images held back to measure progress without training on them  |\
"""))

CELLS.append(code_cell("hd-21", """\
# Cell 8: Compile and train the model.

def compile_and_train(model, x_train, y_train, epochs=5):
    \"\"\"
    Compile the model and train it on the training data.

    Args:
        model   (tf.keras.Sequential): The uncompiled neural-network model.
        x_train (np.ndarray):          Normalised training images.
        y_train (np.ndarray):          Integer digit labels for training.
        epochs  (int):                 Number of passes through the training data.

    Returns:
        history (tf.keras.callbacks.History):
            Object recording loss and accuracy for every epoch.
            Use it to plot learning curves afterwards.
    \"\"\"
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,   # Hold 10 % back for validation
        verbose=1,
    )
    return history


history = compile_and_train(model, x_train, y_train, epochs=5)\
"""))

CELLS.append(md_cell("hd-22", """\
### Inference — Training Output

After each epoch you see two key numbers:

- **loss** / **val_loss**: How wrong the model is.
  → Both should **decrease** as training progresses.
- **accuracy** / **val_accuracy**: Percentage of correct predictions.
  → Both should **increase** as training progresses.

If `val_accuracy` is much lower than `accuracy`, the model may be
**overfitting** (memorising training data instead of learning general patterns).\
"""))

# ── 9. Training history ─────────────────────────────────────────────────
CELLS.append(md_cell("hd-23", """\
## Step 7 — Visualise Training Progress

### What is happening here?
We plot how accuracy and loss changed over each epoch.
This is one of the most important diagnostic charts in machine learning.\
"""))

CELLS.append(code_cell("hd-24", """\
# Cell 9: Plot accuracy and loss curves from the training history.

def plot_training_history(history):
    \"\"\"
    Draw two side-by-side charts:
      - Left:  Training vs validation accuracy over epochs.
      - Right: Training vs validation loss over epochs.

    Good training looks like:
      - Accuracy going UP and getting close to 1.0.
      - Loss going DOWN and getting close to 0.

    Args:
        history (tf.keras.callbacks.History):
            Returned by model.fit(); contains per-epoch metrics.
    \"\"\"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # --- Accuracy chart ---
    ax1.plot(epochs_range, history.history["accuracy"],     marker="o", label="Train Accuracy")
    ax1.plot(epochs_range, history.history["val_accuracy"], marker="s", label="Validation Accuracy")
    ax1.set_title("Accuracy Over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True)

    # --- Loss chart ---
    ax2.plot(epochs_range, history.history["loss"],     marker="o", color="orange", label="Train Loss")
    ax2.plot(epochs_range, history.history["val_loss"], marker="s", color="red",    label="Validation Loss")
    ax2.set_title("Loss Over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Training History — How the Model Improved Each Epoch", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


plot_training_history(history)\
"""))

CELLS.append(md_cell("hd-25", """\
### Inference — Training Charts

| What to look for        | What it means                                                    |
|-------------------------|------------------------------------------------------------------|
| Accuracy rising steeply | The model is learning quickly — good!                            |
| Accuracy flattening out | The model has learnt most of what it can — normal after a few epochs |
| Train ≈ Validation      | The model generalises well — **ideal**                           |
| Train >> Validation     | **Overfitting** — the model memorises training data              |
| Loss falling smoothly   | Learning is stable and progressing correctly                     |\
"""))

# ── 10. Evaluate ────────────────────────────────────────────────────────
CELLS.append(md_cell("hd-26", """\
## Step 8 — Evaluate on the Test Set

### What is happening here?
We feed the **10,000 test images** (never seen during training) to the model
and measure how many it gets right.

### Why this matters
Training accuracy is optimistic — the model has already "seen" those images.
Test accuracy is the **honest** measure of real-world performance.\
"""))

CELLS.append(code_cell("hd-27", """\
# Cell 10: Evaluate the trained model on unseen test data.

def evaluate_model(model, x_test, y_test):
    \"\"\"
    Compute test-set loss and accuracy for a trained Keras model.

    Args:
        model  (tf.keras.Sequential): Trained model.
        x_test (np.ndarray):          Normalised test images.
        y_test (np.ndarray):          True digit labels for the test images.

    Returns:
        test_loss     (float): Average prediction error on the test set.
        test_accuracy (float): Fraction of correct predictions (0.0 – 1.0).
    \"\"\"
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 40)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
    print("=" * 40)
    return test_loss, test_accuracy


test_loss, test_accuracy = evaluate_model(model, x_test, y_test)\
"""))

CELLS.append(md_cell("hd-28", """\
### Inference — Test Accuracy

- A test accuracy around **97–98 %** is typical for this simple two-hidden-layer network.
- That means the model correctly identifies roughly **97 out of every 100** handwritten digits.
- The remaining ~2–3 % are often genuinely hard examples where even humans might hesitate.
- **Loss** measures the average "wrongness" — the closer to 0, the better.\
"""))

# ── 11. Confusion matrix ────────────────────────────────────────────────
CELLS.append(md_cell("hd-29", """\
## Step 9 — Confusion Matrix

### What is a confusion matrix?

It is a 10 × 10 grid that shows **exactly which digits the model mixes up**.

- **Rows** = the actual (true) digit
- **Columns** = the digit the model predicted
- **Diagonal cells** = correct predictions (the model got it right)
- **Off-diagonal cells** = mistakes (the model confused one digit for another)\
"""))

CELLS.append(code_cell("hd-30", """\
# Cell 11: Compute predictions and draw the confusion matrix.

def compute_predictions(model, x_test):
    \"\"\"
    Run the model on all test images and convert probability outputs to labels.

    The model's final layer outputs 10 probabilities (one per digit).
    np.argmax picks the digit with the highest probability.

    Args:
        model  (tf.keras.Sequential): Trained model.
        x_test (np.ndarray):          Normalised test images.

    Returns:
        y_pred (np.ndarray): Predicted digit label for each test image.
    \"\"\"
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred


def plot_confusion_matrix_heatmap(y_test, y_pred):
    \"\"\"
    Draw a colour-coded confusion matrix heatmap and print a
    full classification report.

    Args:
        y_test (np.ndarray): True digit labels.
        y_pred (np.ndarray): Predicted digit labels.
    \"\"\"
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title("Confusion Matrix — Predicted vs Actual Digit", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Digit", fontsize=12)
    plt.ylabel("True Digit", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\\nClassification Report (Precision / Recall / F1 per digit):")
    print(classification_report(y_test, y_pred))


y_pred = compute_predictions(model, x_test)
plot_confusion_matrix_heatmap(y_test, y_pred)\
"""))

CELLS.append(md_cell("hd-31", """\
### Inference — Confusion Matrix

- **Bright diagonal** = the model is doing well for those digits.
- **Bright off-diagonal cell** = a common mistake. For example, if row 4, column 9 is bright,
  the model sometimes confuses the digit **4** for a **9**.
- Digits **1** and **7** are sometimes mixed up (similar appearance).
- Digits **3**, **5**, and **8** can be confused with each other.

### Understanding the Classification Report

| Metric    | What it means                                                          |
|-----------|------------------------------------------------------------------------|
| Precision | Of all images predicted as digit X, what fraction were actually X?     |
| Recall    | Of all images that are truly digit X, what fraction did we find?       |
| F1-score  | A single number combining precision and recall (higher = better)       |
| Support   | How many test images belong to this digit class                        |\
"""))

# ── 12. Sample predictions ──────────────────────────────────────────────
CELLS.append(md_cell("hd-32", """\
## Step 10 — Show Sample Predictions

### What is happening here?
We display 10 test images and compare what the model predicted
against the correct answer.\
"""))

CELLS.append(code_cell("hd-33", """\
# Cell 12: Show 10 test images with true and predicted labels.

def show_predictions(x_test, y_test, y_pred, num_samples=10):
    \"\"\"
    Display test images with colour-coded true vs predicted labels.

    - Green title  → model got it RIGHT  ✓
    - Red title    → model got it WRONG  ✗

    Args:
        x_test      (np.ndarray): Normalised test images.
        y_test      (np.ndarray): True labels.
        y_pred      (np.ndarray): Predicted labels from the model.
        num_samples (int):        How many images to show (max 10 per row).
    \"\"\"
    cols = min(num_samples, 10)
    plt.figure(figsize=(14, 4))
    for i in range(cols):
        plt.subplot(1, cols, i + 1)
        plt.imshow(x_test[i], cmap="gray")
        is_correct = y_pred[i] == y_test[i]
        color = "green" if is_correct else "red"
        mark  = "✓" if is_correct else "✗"
        plt.title(
            f"True: {y_test[i]}\\nPred: {y_pred[i]} {mark}",
            color=color,
            fontsize=9,
        )
        plt.axis("off")
    plt.suptitle(
        "Predictions — Green = Correct  |  Red = Wrong",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


show_predictions(x_test, y_test, y_pred, num_samples=10)\
"""))

CELLS.append(code_cell("hd-34", """\
# Cell 13: Find and display a few images the model got WRONG.

def show_wrong_predictions(x_test, y_test, y_pred, num_samples=10):
    \"\"\"
    Display examples where the model made an incorrect prediction.

    Seeing failures helps us understand the model's weaknesses.

    Args:
        x_test      (np.ndarray): Test images.
        y_test      (np.ndarray): True labels.
        y_pred      (np.ndarray): Predicted labels.
        num_samples (int):        How many wrong examples to display.
    \"\"\"
    wrong_indices = np.where(y_pred != y_test)[0]
    print(f"Total wrong predictions: {len(wrong_indices)} out of {len(y_test)}")

    cols = min(num_samples, len(wrong_indices))
    plt.figure(figsize=(14, 4))
    for i, idx in enumerate(wrong_indices[:cols]):
        plt.subplot(1, cols, i + 1)
        plt.imshow(x_test[idx], cmap="gray")
        plt.title(
            f"True: {y_test[idx]}\\nPred: {y_pred[idx]}",
            color="red",
            fontsize=9,
        )
        plt.axis("off")
    plt.suptitle("Examples the Model Got Wrong", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


show_wrong_predictions(x_test, y_test, y_pred, num_samples=10)\
"""))

CELLS.append(md_cell("hd-35", """\
### Inference — Predictions

- Most of the first 10 test images should have **green (correct)** titles.
- The "wrong" predictions grid shows images that even the model found difficult.
- Look closely — many of these are **genuinely ambiguous handwriting** that a human
  might also second-guess.
- A model that makes ~2–3 % mistakes on this dataset is considered very good
  for such a simple architecture.\
"""))

# ── 13. Save / Load ─────────────────────────────────────────────────────
CELLS.append(md_cell("hd-36", """\
## Step 11 — Save and Reload the Model

### What is happening here?
We save the trained model to disk and then reload it to verify
the saved version gives identical accuracy.

### Why does this matter?
- Training takes time and computing power.
- Once trained, you can **save the model once** and load it later without
  retraining — much like saving a document.
- You can also share the saved file with others or deploy it to an app.\
"""))

CELLS.append(code_cell("hd-37", """\
# Cell 14: Save the trained model to the artifacts directory.

from pathlib import Path

def save_model_to_file(model, save_path):
    \"\"\"
    Save a Keras model to disk in the native Keras format.

    The saved directory contains all model weights, architecture, and
    training configuration — everything needed to reload and use the model.

    Args:
        model     (tf.keras.Sequential): The trained model to save.
        save_path (str or Path):         Destination path (e.g. 'model.keras').
    \"\"\"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved successfully to: {save_path}")


MODEL_PATH = "../artifacts/models/handwritten_digits_model.keras"
save_model_to_file(model, MODEL_PATH)\
"""))

CELLS.append(code_cell("hd-38", """\
# Cell 15: Reload the saved model and verify it produces the same accuracy.

def load_model_from_file(load_path):
    \"\"\"
    Load a previously saved Keras model from disk.

    Args:
        load_path (str or Path): Path to the saved model file.

    Returns:
        loaded_model (tf.keras.Sequential): The restored model, ready to use.
    \"\"\"
    loaded_model = tf.keras.models.load_model(load_path)
    print(f"Model loaded successfully from: {load_path}")
    return loaded_model


loaded_model = load_model_from_file(MODEL_PATH)

# Verify the loaded model gives the same results
loaded_loss, loaded_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"\\nLoaded model — Test Accuracy: {loaded_accuracy * 100:.2f}%")
print("Accuracy matches original model:", abs(loaded_accuracy - test_accuracy) < 1e-6)\
"""))

CELLS.append(md_cell("hd-39", """\
### Inference — Save and Load

- The loaded model should report **exactly the same accuracy** as the original.
- This confirms the file was saved correctly and nothing was lost.
- The `.keras` format stores everything: weights, architecture, and optimizer state.\
"""))

# ── 14. Single image prediction ─────────────────────────────────────────
CELLS.append(md_cell("hd-40", """\
## Bonus — Predict a Single New Image

### What is happening here?
We pick one test image, feed it to the model, and see exactly what
the model "thinks" — including its confidence for every digit.\
"""))

CELLS.append(code_cell("hd-41", """\
# Cell 16: Predict a single image and display the confidence scores.

def predict_single_image(model, image, true_label):
    \"\"\"
    Predict the digit in a single image and display confidence for all classes.

    The model outputs 10 probabilities. The highest one is the prediction.

    Args:
        model      (tf.keras.Sequential): Trained model.
        image      (np.ndarray):          A single 28×28 normalised image.
        true_label (int):                 The actual digit shown in the image.
    \"\"\"
    # The model expects a batch, so we add a dimension: (28,28) → (1,28,28)
    image_batch = np.expand_dims(image, axis=0)
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_digit = np.argmax(probabilities)

    # ── Show the image ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.imshow(image, cmap="gray")
    color = "green" if predicted_digit == true_label else "red"
    ax1.set_title(
        f"True: {true_label}   Predicted: {predicted_digit}",
        fontsize=13,
        fontweight="bold",
        color=color,
    )
    ax1.axis("off")

    # ── Show the confidence bar chart ───────────────────────────────
    colors = ["steelblue"] * 10
    colors[predicted_digit] = "green" if predicted_digit == true_label else "red"
    ax2.bar(range(10), probabilities, color=colors, edgecolor="black")
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit (0–9)", fontsize=11)
    ax2.set_ylabel("Confidence (Probability)", fontsize=11)
    ax2.set_title("Model Confidence for Each Digit", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.4)

    plt.suptitle("Single Image Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"Predicted: {predicted_digit}  |  True: {true_label}  |  Confidence: {probabilities[predicted_digit]*100:.1f}%")


# Use the 7th test image as an example
predict_single_image(loaded_model, x_test[6], y_test[6])\
"""))

CELLS.append(md_cell("hd-42", """\
### Inference — Single Prediction

- The **tall bar** shows which digit the model is most confident about.
- A healthy prediction looks like: one very tall bar and all others near zero.
- If two bars are almost the same height, the model is **uncertain** — this
  often happens with genuinely ambiguous handwriting.\
"""))

# ── 15. Final summary ────────────────────────────────────────────────────
CELLS.append(md_cell("hd-43", """\
## Final Summary

### What we built and learnt

| Step | What we did                          | Key concept                              |
|------|--------------------------------------|------------------------------------------|
| 1    | Loaded MNIST                         | 60k train / 10k test images              |
| 2    | Explored class distribution          | Dataset is balanced (≈6,000 per digit)   |
| 3    | Visualised sample images             | 28×28 greyscale pixels per image         |
| 4    | Normalised pixel values              | Divide by 255 → range 0.0–1.0            |
| 5    | Built a Sequential neural network    | Flatten → Dense(128) → Dense(128) → Dense(10, softmax) |
| 6    | Compiled and trained (5 epochs)      | Adam optimiser, crossentropy loss        |
| 7    | Plotted training history             | Accuracy↑ and loss↓ confirm learning     |
| 8    | Evaluated on test set                | ~97–98 % accuracy on unseen data         |
| 9    | Confusion matrix                     | Shows exactly which digits get mixed up  |
| 10   | Displayed correct & wrong preds      | Most mistakes are on ambiguous images    |
| 11   | Saved and reloaded model             | `.keras` format preserves everything     |
| 12   | Single image prediction              | Confidence bar shows model certainty     |

---

### Key Takeaways

1. **Data preparation matters** — normalising is a small step that makes a big difference.
2. **More epochs ≠ always better** — too many can cause overfitting.
3. **Test accuracy is the honest metric** — training accuracy is too optimistic.
4. **Confusion matrix** is more informative than a single accuracy number.
5. **Saving models** means you never have to train from scratch again.

---

### What could you try next?

- Add a **Dropout layer** (e.g. `tf.keras.layers.Dropout(0.2)`) to reduce overfitting.
- Try **more epochs** (10–20) and see if accuracy improves.
- Use a **Convolutional Neural Network (CNN)** — it reads images the way eyes do and typically reaches 99 %+.
- Try the model on your **own handwritten digit** (photograph, resize to 28×28, invert colours).

---

*Notebook authored for the machine-learning-project — feature/handwritten-digit-classification branch.*\
"""))

# ──────────────────────────────────────────────────────────────────────
# Assemble and write the notebook
# ──────────────────────────────────────────────────────────────────────

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (Python 3.13)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).resolve().parents[1] / "notebooks" / "handwritten_digits_tensorflow_step_by_step.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(CELLS)}")
