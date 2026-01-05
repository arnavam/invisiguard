"""
Rule-Based Motion Detection: Predict on a single file
Produces output similar to predict_single_file.py but using rule-based detection.

Outputs:
- Console summary with per-window predictions
- CSV file with detailed results
- Plot with confidence overlay (like file_prediction_plot.png)
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from detect_motion_rulebased import SuddenMotionDetector


def select_numeric_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
    if cols:
        return [c for c in cols if c in df.columns]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def predict_file_rulebased(
    csv_path: str,
    seq_len: int = 100,
    step_size: int = 20,
    cols: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_csv: str = "rulebased_predictions.csv",
    plot_cols: Optional[List[str]] = None,
    save_plot: str = "rulebased_prediction_plot.png",
    window_size: int = 20,
    use_kalman: bool = False,
    gravity: float = 1.0,
):
    """
    Analyze a CSV file using rule-based sudden motion detection.
    
    Similar interface to predict_file() but uses SuddenMotionDetector
    instead of the LSTM model.
    
    Args:
        csv_path: Path to the input CSV file
        seq_len: Sequence length for windowing (samples per window)
        step_size: Step size between windows
        cols: Columns to use [x, y, z] - auto-detected if None
        threshold: Threshold for sudden motion detection (on combined score 0-1)
        save_csv: Path to save detailed CSV results
        plot_cols: Columns to plot (first 2-3 from cols if None)
        save_plot: Path to save the plot
        window_size: Rolling window size for the detector
        use_kalman: Whether to use Kalman filter for noise reduction
        gravity: Set to 9.81 if data is in m/s², or 1.0 if already in g
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    selected = select_numeric_columns(df, cols)
    if len(selected) < 3:
        raise ValueError(f"Need at least 3 numeric columns for x,y,z. Found: {selected}")
    
    # Use first 3 columns as x, y, z
    x_col, y_col, z_col = selected[0], selected[1], selected[2]
    data = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
    
    # Create detector
    detector = SuddenMotionDetector(
        window_size=window_size,
        use_kalman=use_kalman,
        gravity=gravity,
    )
    
    # Run detection on entire file sample-by-sample
    sample_results = detector.detect_batch(data)
    
    # Build sliding windows and aggregate per-window probabilities
    n_samples = len(data)
    n_windows = max(1, (n_samples - seq_len) // step_size + 1)
    
    window_ranges = []
    window_probs = []
    window_preds = []
    window_sudden_counts = []
    
    for i in range(n_windows):
        start = i * step_size
        end = min(start + seq_len - 1, n_samples - 1)
        window_ranges.append((start, end))
        
        # Get sample results in this window
        window_samples = sample_results[start:end+1]
        
        # Aggregate: use average confidence of sudden samples, or overall "suddenness" score
        # We'll compute the fraction of samples that are sudden + average confidence
        sudden_count = sum(1 for s in window_samples if s.is_sudden)
        total = len(window_samples)
        
        if total == 0:
            prob = 0.0
        else:
            # Combine: fraction sudden + average confidence of sudden samples
            sudden_samples = [s for s in window_samples if s.is_sudden]
            if sudden_samples:
                avg_confidence = np.mean([s.confidence for s in sudden_samples])
                # Probability = weighted combination
                prob = 0.5 * (sudden_count / total) + 0.5 * avg_confidence
            else:
                # No sudden samples - use inverse of average confidence for normal
                normal_confidence = np.mean([s.confidence for s in window_samples])
                prob = max(0, 1 - normal_confidence) * 0.5
        
        window_probs.append(prob)
        window_preds.append(1 if prob >= threshold else 0)
        window_sudden_counts.append(sudden_count)
    
    probs = np.array(window_probs)
    preds = np.array(window_preds)
    
    # Summary
    n_sudden = int(preds.sum())
    n_normal = n_windows - n_sudden
    
    print(f"File: {csv_path}")
    print(f"Using columns: {[x_col, y_col, z_col]}")
    print(f"Seq len: {seq_len}, step: {step_size}, windows: {n_windows}")
    print(f"Predicted sudden windows: {n_sudden}, normal: {n_normal}")
    
    if n_sudden == n_windows:
        print("SUMMARY: All windows predict SUDDEN MOTION")
    elif n_sudden == 0:
        print("SUMMARY: All windows predict NORMAL MOTION")
    else:
        print("SUMMARY: Mixed predictions across windows")
    
    # Print per-window short report
    print("\nWindow index | start:end | prob | pred | sudden_samples")
    for i, (s, e) in enumerate(window_ranges):
        pred_label = 'SUDDEN' if preds[i] == 1 else 'NORMAL'
        print(f"{i:03d} | {s:04d}:{e:04d} | {probs[i]:.3f} | {pred_label} | {window_sudden_counts[i]}")
    
    # Save detailed CSV
    results = pd.DataFrame({
        "window_index": np.arange(n_windows),
        "start": [r[0] for r in window_ranges],
        "end": [r[1] for r in window_ranges],
        "probability": probs,
        "prediction": preds,
        "sudden_sample_count": window_sudden_counts,
    })
    results.to_csv(save_csv, index=False)
    print(f"\nDetailed per-window results saved to: {os.path.abspath(save_csv)}")
    
    # Also save per-sample results
    sample_csv = save_csv.replace(".csv", "_samples.csv")
    sample_df = pd.DataFrame({
        "index": np.arange(len(sample_results)),
        "is_sudden": [r.is_sudden for r in sample_results],
        "motion_type": [r.motion_type for r in sample_results],
        "confidence": [r.confidence for r in sample_results],
        "magnitude": [r.magnitude for r in sample_results],
        "jerk": [r.jerk for r in sample_results],
        "rolling_std": [r.rolling_std for r in sample_results],
        "score_magnitude": [r.score_magnitude for r in sample_results],
        "score_jerk": [r.score_jerk for r in sample_results],
        "score_rolling_std": [r.score_rolling_std for r in sample_results],
    })
    sample_df.to_csv(sample_csv, index=False)
    print(f"Per-sample results saved to: {os.path.abspath(sample_csv)}")
    
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
        ax.set_title(f"Rule-Based Motion Detection: {plot_cols} with probability overlay (red intensity ~ prob)")
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
    csv_path = r"JO_FALL/volunteer_1_left_hand/fall/Fall_backwards2.csv"  # path to your CSV
    seq_len = 100
    step_size = 20
    cols = ["Acc_x", "Acc_y", "Acc_z"]  # or None to auto-select numeric cols
    threshold = 0.5
    save_csv = "rulebased_predictions.csv"
    save_plot = "rulebased_prediction_plot.png"
    
    # Detector settings
    window_size = 20       # Rolling window for detector
    use_kalman = False     # Set True for noisy data
    gravity = 1.0          # Set to 9.81 if data is in m/s²
    
    predict_file_rulebased(
        csv_path=csv_path,
        seq_len=seq_len,
        step_size=step_size,
        cols=cols,
        threshold=threshold,
        save_csv=save_csv,
        save_plot=save_plot,
        window_size=window_size,
        use_kalman=use_kalman,
        gravity=gravity,
    )
