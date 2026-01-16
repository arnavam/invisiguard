"""
Encoder-Decoder Based Motion Detection: Predict on a single file

Uses LSTMAutoencoder or LSTMEventClassifier from lstm_encoder_decoder.py
to detect events (e.g., falls, clapping, sudden motion) in sensor data.

Two detection modes:
1. Autoencoder mode: Uses reconstruction error to detect anomalies/events
2. Classifier mode: Direct supervised classification

Outputs:
- Console summary with per-window predictions
- CSV file with detailed results
- Plot with probability/error overlay
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from lstm_encoder_decoder import LSTMAutoencoder, LSTMEventClassifier


def select_numeric_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
    """Select numeric columns from dataframe."""
    if cols:
        return [c for c in cols if c in df.columns]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def create_sliding_windows(data: np.ndarray, seq_len: int = 100, step_size: int = 20) -> List[np.ndarray]:
    """Create sliding windows from data array."""
    windows = []
    for i in range(0, len(data) - seq_len + 1, step_size):
        windows.append(data[i:i + seq_len])
    return windows


def normalize_sequences(sequences: np.ndarray) -> np.ndarray:
    """Normalize sequences per-feature across all windows."""
    n_windows, seq_len, n_features = sequences.shape
    
    # Flatten to (n_windows * seq_len, n_features), normalize, reshape back
    flat = sequences.reshape(-1, n_features)
    
    # Normalize each feature to [0, 1]
    for f in range(n_features):
        col = flat[:, f]
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 1e-8:
            flat[:, f] = (col - min_val) / (max_val - min_val)
        else:
            flat[:, f] = 0.0
    
    return flat.reshape(n_windows, seq_len, n_features)


def predict_file_autoencoder(
    csv_path: str,
    ckpt_path: str,
    seq_len: int = 100,
    step_size: int = 20,
    cols: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    save_csv: str = "encoder_decoder_predictions.csv",
    plot_cols: Optional[List[str]] = None,
    save_plot: str = "encoder_decoder_prediction_plot.png",
) -> pd.DataFrame:
    """
    Analyze a CSV file using the autoencoder-based event detector.
    
    Detects events based on reconstruction error. High error = event detected.
    
    Args:
        csv_path: Path to the input CSV file
        ckpt_path: Path to the trained autoencoder checkpoint (.ckpt)
        seq_len: Sequence length for windowing (samples per window)
        step_size: Step size between windows
        cols: Columns to use [x, y, z] - auto-detected if None
        threshold: Threshold for event detection (auto-computed if None)
        save_csv: Path to save detailed CSV results
        plot_cols: Columns to plot (first 3 from cols if None)
        save_plot: Path to save the plot
        
    Returns:
        DataFrame with per-window results
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    selected = select_numeric_columns(df, cols)
    if len(selected) < 3:
        raise ValueError(f"Need at least 3 numeric columns for x,y,z. Found: {selected}")
    
    # Use first 3 columns as x, y, z
    x_col, y_col, z_col = selected[0], selected[1], selected[2]
    data = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
    
    # Create sliding windows
    windows = create_sliding_windows(data, seq_len=seq_len, step_size=step_size)
    sequences = np.array(windows, dtype=np.float32)  # (n_windows, seq_len, 3)
    
    # Normalize
    sequences = normalize_sequences(sequences)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        inputs = torch.from_numpy(sequences).float().to(device)
        reconstructed, latent = model(inputs)
        
        # Compute reconstruction errors
        errors = model.compute_reconstruction_error(inputs, reconstructed)
        errors = errors.cpu().numpy()
        
        # Determine threshold if not provided
        if threshold is None:
            # Use model's threshold or compute from data
            threshold = model.reconstruction_threshold
            if threshold <= 0:
                # Use mean + 2*std as default threshold
                threshold = float(errors.mean() + 2 * errors.std())
        
        # Predictions based on error > threshold
        preds = (errors > threshold).astype(int)
        
        # Normalize errors for probability-like scores (0-1)
        error_min, error_max = errors.min(), errors.max()
        if error_max - error_min > 1e-8:
            probs = (errors - error_min) / (error_max - error_min)
        else:
            probs = np.zeros_like(errors)
    
    # Map windows to sample ranges
    window_ranges = []
    for i in range(len(sequences)):
        start = i * step_size
        end = start + seq_len - 1
        if end >= len(data):
            end = len(data) - 1
        window_ranges.append((start, end))
    
    # Summary
    n_windows = len(sequences)
    n_event = int(preds.sum())
    n_normal = n_windows - n_event
    
    print(f"File: {csv_path}")
    print(f"Using columns: {[x_col, y_col, z_col]}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Seq len: {seq_len}, step: {step_size}, windows: {n_windows}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Predicted event windows: {n_event}, normal: {n_normal}")
    
    if n_event == n_windows:
        print("SUMMARY: All windows predict EVENT/ANOMALY")
    elif n_event == 0:
        print("SUMMARY: All windows predict NORMAL")
    else:
        print("SUMMARY: Mixed predictions across windows")
    
    # Print per-window short report
    print("\nWindow index | start:end | error | norm_score | pred")
    for i, (s, e) in enumerate(window_ranges):
        pred_label = 'EVENT' if preds[i] == 1 else 'NORMAL'
        print(f"{i:03d} | {s:04d}:{e:04d} | {errors[i]:.6f} | {probs[i]:.3f} | {pred_label}")
    
    # Save detailed CSV
    results = pd.DataFrame({
        "window_index": np.arange(n_windows),
        "start": [r[0] for r in window_ranges],
        "end": [r[1] for r in window_ranges],
        "reconstruction_error": errors,
        "normalized_score": probs,
        "prediction": preds,
    })
    results.to_csv(save_csv, index=False)
    print(f"\nDetailed per-window results saved to: {os.path.abspath(save_csv)}")
    
    # ------- Plot -------
    _create_plot(df, data, window_ranges, probs, errors, plot_cols, x_col, y_col, z_col, save_plot, mode="autoencoder")
    
    return results


def predict_file_classifier(
    csv_path: str,
    ckpt_path: str,
    seq_len: int = 100,
    step_size: int = 20,
    cols: Optional[List[str]] = None,
    save_csv: str = "classifier_predictions.csv",
    plot_cols: Optional[List[str]] = None,
    save_plot: str = "classifier_prediction_plot.png",
    label_names: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Analyze a CSV file using the classifier-based event detector.
    
    Direct classification into event categories.
    
    Args:
        csv_path: Path to the input CSV file
        ckpt_path: Path to the trained classifier checkpoint (.ckpt)
        seq_len: Sequence length for windowing (samples per window)
        step_size: Step size between windows
        cols: Columns to use [x, y, z] - auto-detected if None
        save_csv: Path to save detailed CSV results
        plot_cols: Columns to plot (first 3 from cols if None)
        save_plot: Path to save the plot
        label_names: Optional mapping from class index to name
        
    Returns:
        DataFrame with per-window results
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    selected = select_numeric_columns(df, cols)
    if len(selected) < 3:
        raise ValueError(f"Need at least 3 numeric columns for x,y,z. Found: {selected}")
    
    # Use first 3 columns as x, y, z
    x_col, y_col, z_col = selected[0], selected[1], selected[2]
    data = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
    
    # Create sliding windows
    windows = create_sliding_windows(data, seq_len=seq_len, step_size=step_size)
    sequences = np.array(windows, dtype=np.float32)  # (n_windows, seq_len, 3)
    
    # Normalize
    sequences = normalize_sequences(sequences)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEventClassifier.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Default label names
    if label_names is None:
        label_names = {0: "normal/adl", 1: "event/fall"}
    
    # Inference
    with torch.no_grad():
        inputs = torch.from_numpy(sequences).float().to(device)
        preds, confidences = model.predict(inputs)
        preds = preds.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # Get probability of event class (assuming class 1 is event)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Prob of class 1
    
    # Map windows to sample ranges
    window_ranges = []
    for i in range(len(sequences)):
        start = i * step_size
        end = start + seq_len - 1
        if end >= len(data):
            end = len(data) - 1
        window_ranges.append((start, end))
    
    # Summary
    n_windows = len(sequences)
    unique, counts = np.unique(preds, return_counts=True)
    
    print(f"File: {csv_path}")
    print(f"Using columns: {[x_col, y_col, z_col]}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Seq len: {seq_len}, step: {step_size}, windows: {n_windows}")
    print(f"Class distribution:")
    for u, c in zip(unique, counts):
        name = label_names.get(u, f"class_{u}")
        print(f"  {name}: {c} ({100*c/n_windows:.1f}%)")
    
    # Print per-window short report
    print("\nWindow index | start:end | confidence | event_prob | pred")
    for i, (s, e) in enumerate(window_ranges):
        pred_label = label_names.get(preds[i], f"class_{preds[i]}")
        print(f"{i:03d} | {s:04d}:{e:04d} | {confidences[i]:.3f} | {probs[i]:.3f} | {pred_label}")
    
    # Save detailed CSV
    results = pd.DataFrame({
        "window_index": np.arange(n_windows),
        "start": [r[0] for r in window_ranges],
        "end": [r[1] for r in window_ranges],
        "predicted_class": preds,
        "predicted_label": [label_names.get(p, f"class_{p}") for p in preds],
        "confidence": confidences,
        "event_probability": probs,
    })
    results.to_csv(save_csv, index=False)
    print(f"\nDetailed per-window results saved to: {os.path.abspath(save_csv)}")
    
    # ------- Plot -------
    _create_plot(df, data, window_ranges, probs, None, plot_cols, x_col, y_col, z_col, save_plot, mode="classifier")
    
    return results


def _create_plot(
    df: pd.DataFrame,
    data: np.ndarray,
    window_ranges: List[Tuple[int, int]],
    probs: np.ndarray,
    errors: Optional[np.ndarray],
    plot_cols: Optional[List[str]],
    x_col: str,
    y_col: str,
    z_col: str,
    save_plot: str,
    mode: str = "autoencoder",
) -> None:
    """Create and save the visualization plot."""
    if plot_cols is None:
        plot_cols = [x_col, y_col, z_col]
    
    plot_cols = [c for c in plot_cols if c in df.columns]
    if len(plot_cols) == 0:
        print("No valid plot columns found; skipping plot.")
        return
    
    time = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot first 2-3 columns
    ax.plot(time, df[plot_cols[0]].to_numpy(dtype=float), label=plot_cols[0], linewidth=1.2)
    if len(plot_cols) > 1 and plot_cols[1] != plot_cols[0]:
        ax.plot(time, df[plot_cols[1]].to_numpy(dtype=float), label=plot_cols[1], linewidth=1.0, alpha=0.9)
    if len(plot_cols) > 2 and plot_cols[2] not in [plot_cols[0], plot_cols[1]]:
        ax.plot(time, df[plot_cols[2]].to_numpy(dtype=float), label=plot_cols[2], linewidth=0.8, alpha=0.8)
    
    # Overlay probability as semi-transparent red spans where intensity ~ prob
    max_alpha = 0.7
    for i, (s, e) in enumerate(window_ranges):
        p = float(probs[i])
        if p <= 0:
            continue
        alpha = max_alpha * p
        ax.axvspan(s, e, color="red", alpha=alpha, linewidth=0)
    
    # Plot probability/error curve on secondary y-axis
    midpoints = np.array([(s + e) / 2 for s, e in window_ranges])
    ax2 = ax.twinx()
    
    if mode == "autoencoder" and errors is not None:
        ax2.plot(midpoints, errors, color="purple", linestyle="--", marker="o", markersize=3, label="Reconstruction error")
        ax2.set_ylabel("Reconstruction Error", color="purple")
    else:
        ax2.plot(midpoints, probs, color="black", linestyle="--", marker="o", markersize=3, label="Event probability")
        ax2.set_ylabel("Event Probability")
    
    ax2.set_ylim(0, max(probs.max() * 1.1, 1.0) if errors is None else errors.max() * 1.1)
    
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Acceleration Value")
    
    title_mode = "Autoencoder" if mode == "autoencoder" else "Classifier"
    ax.set_title(f"LSTM {title_mode} Event Detection: {plot_cols} (red intensity ~ event score)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    
    try:
        plt.savefig(save_plot, dpi=200, bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(save_plot)}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    plt.show()


if __name__ == "__main__":
    # Configure these variables (similar to predict_single_file_ml.py)
    csv_path = r"JO_FALL/volunteer_1_left_hand/adl/applaud.csv"  # path to your CSV
    
    # Choose detection mode: "autoencoder" or "classifier"
    detection_mode = "autoencoder"  # or "classifier"
    
    # Path to trained checkpoint
    # NOTE: You need to train the model first using lstm_encoder_decoder.py
    if detection_mode == "autoencoder":
        ckpt_path = r"checkpoints/autoencoder/lstm_autoencoder-epoch=09-val_loss=0.0100.ckpt"
    else:
        ckpt_path = r"checkpoints/classifier/lstm_classifier-epoch=09-val_f1=0.9500.ckpt"
    
    seq_len = 100
    step_size = 20
    cols = ["Acc_x", "Acc_y", "Acc_z"]  # or None to auto-select numeric cols
    
    # Create predictions directory if needed
    os.makedirs("predictions", exist_ok=True)
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print("=" * 60)
        print("ERROR: No trained model checkpoint found!")
        print("=" * 60)
        print(f"\nLooking for: {ckpt_path}")
        print("\nYou need to train the model first:")
        print("  1. Run: python lstm_encoder_decoder.py")
        print("  2. This will train both autoencoder and classifier")
        print("  3. Checkpoints will be saved in checkpoints/ folder")
        print("\nAlternatively, you can use the rule-based detector:")
        print("  python predict_single_file_rulebased.py")
        print("=" * 60)
    else:
        if detection_mode == "autoencoder":
            predict_file_autoencoder(
                csv_path=csv_path,
                ckpt_path=ckpt_path,
                seq_len=seq_len,
                step_size=step_size,
                cols=cols,
                threshold=None,  # Auto-compute or use model's threshold
                save_csv="predictions/encoder_decoder_predictions.csv",
                save_plot="predictions/encoder_decoder_prediction_plot.png",
            )
        else:
            predict_file_classifier(
                csv_path=csv_path,
                ckpt_path=ckpt_path,
                seq_len=seq_len,
                step_size=step_size,
                cols=cols,
                save_csv="predictions/classifier_predictions.csv",
                save_plot="predictions/classifier_prediction_plot.png",
                label_names={0: "adl", 1: "fall"},  # Adjust based on your label_map
            )
