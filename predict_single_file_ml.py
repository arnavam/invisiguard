"""
ML-Based Motion Detection: Predict on a single file
Produces output similar to predict_single_file.py but using the ML motion classifier.

Uses MotionClassifierLSTM from detect_motion_ml.py to classify sudden vs normal motion.

Outputs:
- Console summary with per-window predictions
- CSV file with detailed results
- Plot with probability overlay (like file_prediction_plot.png)
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from detect_motion_ml import MotionClassifierLSTM, MotionFeatureExtractor


def select_numeric_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
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
    # Shape: (n_windows, seq_len, n_features)
    n_windows, seq_len, n_features = sequences.shape
    
    # Flatten to (n_windows * seq_len, n_features), normalize, reshape back
    flat = sequences.reshape(-1, n_features)
    
    # Normalize each feature to [0, 1] or z-score
    for f in range(n_features):
        col = flat[:, f]
        min_val, max_val = col.min(), col.max()
        if max_val - min_val > 1e-8:
            flat[:, f] = (col - min_val) / (max_val - min_val)
        else:
            flat[:, f] = 0.0
    
    return flat.reshape(n_windows, seq_len, n_features)


def predict_file_ml(
    csv_path: str,
    ckpt_path: str,
    seq_len: int = 100,
    step_size: int = 20,
    cols: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_csv: str = "ml_motion_predictions.csv",
    plot_cols: Optional[List[str]] = None,
    save_plot: str = "ml_motion_prediction_plot.png",
    use_features: bool = True,
):
    """
    Analyze a CSV file using the ML-based motion classifier.
    
    Args:
        csv_path: Path to the input CSV file
        ckpt_path: Path to the trained model checkpoint (.ckpt)
        seq_len: Sequence length for windowing (samples per window)
        step_size: Step size between windows
        cols: Columns to use [x, y, z] - auto-detected if None
        threshold: Threshold for sudden motion classification (0-1)
        save_csv: Path to save detailed CSV results
        plot_cols: Columns to plot (first 2-3 from cols if None)
        save_plot: Path to save the plot
        use_features: Whether to extract motion features (must match training)
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
    
    # Extract features if needed
    if use_features:
        sequences = MotionFeatureExtractor.extract_batch(sequences)
    
    # Normalize
    sequences = normalize_sequences(sequences)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionClassifierLSTM.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        inputs = torch.from_numpy(sequences).float().to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)
    
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
    n_sudden = int(preds.sum())
    n_normal = n_windows - n_sudden
    
    print(f"File: {csv_path}")
    print(f"Using columns: {[x_col, y_col, z_col]}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Seq len: {seq_len}, step: {step_size}, windows: {n_windows}")
    print(f"Predicted sudden windows: {n_sudden}, normal: {n_normal}")
    
    if n_sudden == n_windows:
        print("SUMMARY: All windows predict SUDDEN MOTION")
    elif n_sudden == 0:
        print("SUMMARY: All windows predict NORMAL MOTION")
    else:
        print("SUMMARY: Mixed predictions across windows")
    
    # Print per-window short report
    print("\nWindow index | start:end | prob | pred")
    for i, (s, e) in enumerate(window_ranges):
        pred_label = 'SUDDEN' if preds[i] == 1 else 'NORMAL'
        print(f"{i:03d} | {s:04d}:{e:04d} | {probs[i]:.3f} | {pred_label}")
    
    # Save detailed CSV
    results = pd.DataFrame({
        "window_index": np.arange(n_windows),
        "start": [r[0] for r in window_ranges],
        "end": [r[1] for r in window_ranges],
        "probability": probs,
        "prediction": preds,
    })
    results.to_csv(save_csv, index=False)
    print(f"\nDetailed per-window results saved to: {os.path.abspath(save_csv)}")
    
    # ------- Plot -------
    if plot_cols is None:
        plot_cols = [x_col, y_col, z_col]
    
    plot_cols = [c for c in plot_cols if c in df.columns]
    if len(plot_cols) == 0:
        print("No valid plot columns found; skipping plot.")
    else:
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
        
        # Plot probability curve on secondary y-axis
        midpoints = np.array([(s + e) / 2 for s, e in window_ranges])
        ax2 = ax.twinx()
        ax2.plot(midpoints, probs, color="black", linestyle="--", marker="o", markersize=3, label="Sudden probability")
        ax2.set_ylabel("Sudden Motion Probability")
        ax2.set_ylim(0, 1.05)
        
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Acceleration Value")
        ax.set_title(f"ML Motion Detection: {plot_cols} with probability overlay (red intensity ~ prob)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        
        try:
            plt.savefig(save_plot, dpi=200, bbox_inches="tight")
            print(f"Plot saved to: {os.path.abspath(save_plot)}")
        except Exception as e:
            print(f"Could not save plot: {e}")
        plt.show()
    
    return results


if __name__ == "__main__":
    # Configure these variables (similar to predict_single_file.py)
    csv_path = r"JO_FALL/volunteer_1_left_hand/adl/applaud.csv"  # path to your CSV
    
    # Path to trained motion classifier checkpoint
    # NOTE: You need to train the motion classifier first using detect_motion_ml.py
    # or use the existing fall detection checkpoint if applicable
    ckpt_path = r"checkpoints/motion/motion-classifier-epoch=09-val_loss=0.27.ckpt"
    
    seq_len = 100
    step_size = 20
    cols = ["Acc_x", "Acc_y", "Acc_z"]  # or None to auto-select numeric cols
    threshold = 0.5
    save_csv = "predictions/ml_motion_predictions.csv"
    save_plot = "predictions/ml_motion_prediction_plot.png"
    use_features = True  # Must match how model was trained
    
    # Check if checkpoint exists
    if not os.path.exists(ckpt_path):
        print("=" * 60)
        print("ERROR: No trained motion classifier checkpoint found!")
        print("=" * 60)
        print(f"\nLooking for: {ckpt_path}")
        print("\nYou need to train the motion classifier first:")
        print("  1. Run: python detect_motion_ml.py")
        print("  2. Or train with your own data:")
        print("     from detect_motion_ml import train_motion_classifier")
        print("     model, _, _ = train_motion_classifier(sequences, labels)")
        print("\nAlternatively, you can use the rule-based detector:")
        print("  python predict_single_file_rulebased.py")
        print("=" * 60)
    else:
        predict_file_ml(
            csv_path=csv_path,
            ckpt_path=ckpt_path,
            seq_len=seq_len,
            step_size=step_size,
            cols=cols,
            threshold=threshold,
            save_csv=save_csv,
            save_plot=save_plot,
            use_features=use_features,
        )
