import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from create_dataset import create_sliding_windows_multi, normalize_sequences_per_feature
from train_fall import FallDetectionLSTM


def select_numeric_columns(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
    if cols:
        return [c for c in cols if c in df.columns]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def predict_file(
    csv_path: str,
    ckpt_path: str,
    seq_len: int = 100,
    step_size: int = 20,
    cols: Optional[List[str]] = None,
    threshold: float = 0.5,
    save_csv: str = "file_predictions.csv",
    plot_cols: Optional[List[str]] = None,
    save_plot: str = "file_prediction_plot.png",
):
    # load csv
    df = pd.read_csv(csv_path)
    selected = select_numeric_columns(df, cols)
    if not selected:
        raise ValueError("No numeric columns found / selected in CSV")

    data = df[selected].to_numpy(dtype=np.float32)

    # build sliding windows
    sequences = create_sliding_windows_multi(data, seq_len=seq_len, step_size=step_size)
    sequences = np.array(sequences, dtype=np.float32)  # shape (n_windows, seq_len, n_features)

    # normalize per-feature across windows (same normalizer used in training script)
    sequences = normalize_sequences_per_feature(sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model (Lightning will recreate model from checkpoint hparams)
    model = FallDetectionLSTM.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    # inference
    with torch.no_grad():
        inputs = torch.from_numpy(sequences).float().to(device)
        logits = model(inputs)  # returns shape (n_windows,) or (n_windows,1) squeezed
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= threshold).astype(int)

    # map windows to sample ranges
    window_ranges = []
    for i in range(len(sequences)):
        start = i * step_size
        end = start + seq_len - 1
        if end >= len(data):
            end = len(data) - 1
        window_ranges.append((start, end))

    # summary
    n_windows = len(sequences)
    n_fall = int(preds.sum())
    n_nonfall = n_windows - n_fall

    print(f"File: {csv_path}")
    print(f"Using columns: {selected}")
    print(f"Seq len: {seq_len}, step: {step_size}, windows: {n_windows}")
    print(f"Predicted fall windows: {n_fall}, non-fall: {n_nonfall}")

    if n_fall == n_windows:
        print("SUMMARY: All windows predict FALL")
    elif n_fall == 0:
        print("SUMMARY: All windows predict NON-FALL")
    else:
        print("SUMMARY: Mixed predictions across windows")

    # print per-window short report
    print("\nWindow index | start:end | prob | pred")
    for i, (s, e) in enumerate(window_ranges):
        print(f"{i:03d} | {s:04d}:{e:04d} | {probs[i]:.3f} | {'FALL' if preds[i]==1 else 'NO-FALL'}")

    # save detailed csv
    results = pd.DataFrame(
        {
            "window_index": np.arange(n_windows),
            "start": [r[0] for r in window_ranges],
            "end": [r[1] for r in window_ranges],
            "probability": probs,
            "prediction": preds,
        }
    )
    results.to_csv(save_csv, index=False)
    print(f"\nDetailed per-window results saved to: {os.path.abspath(save_csv)}")

    # ------- plot two features with probability overlay (use probability directly) -------
    # decide which columns to plot
    if plot_cols is None:
        if len(selected) >= 2:
            plot_cols = [selected[0], selected[1]]
        else:
            plot_cols = [selected[0], selected[0]]

    plot_cols = [c for c in plot_cols if c in df.columns]
    if len(plot_cols) == 0:
        print("No valid plot columns found; skipping plot.")
    else:
        time = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(time, df[plot_cols[0]].to_numpy(dtype=float), label=plot_cols[0], linewidth=1.2)
        if len(plot_cols) > 1 and plot_cols[1] != plot_cols[0]:
            ax.plot(time, df[plot_cols[1]].to_numpy(dtype=float), label=plot_cols[1], linewidth=1.0, alpha=0.9)

        # overlay probability as semi-transparent red spans where intensity ~ prob
        max_alpha = 0.7  # scale for visibility
        for i, (s, e) in enumerate(window_ranges):
            p = float(probs[i])
            if p <= 0:
                continue
            alpha = max_alpha * p
            ax.axvspan(s, e, color="red", alpha=alpha, linewidth=0)

        # also plot probability curve (sampled at window midpoints) on a secondary y-axis
        midpoints = np.array([(s + e) / 2 for s, e in window_ranges])
        ax2 = ax.twinx()
        ax2.plot(midpoints, probs, color="black", linestyle="--", marker="o", markersize=3, label="Fall probability")
        ax2.set_ylabel("Fall probability")
        ax2.set_ylim(0, 1.05)

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.set_title(f"Features {plot_cols} with probability overlay (red intensity ~ prob)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        try:
            plt.savefig(save_plot, dpi=200, bbox_inches="tight")
            print(f"Plot saved to: {os.path.abspath(save_plot)}")
        except Exception:
            pass
        plt.show()

    return results

if __name__ == "__main__":
    # Configure these variables instead of using argparse
    csv_path = r"JO_FALL/volunteer_1_left_hand/fall/Fall_backwards2.csv"  # path to your CSV
    ckpt_path = r"checkpoints/fall-detection-epoch=35-val_loss=0.26.ckpt"  # path to your checkpoint
    seq_len = 100
    step_size = 20
    cols = None  # e.g. ["acc_x", "acc_y", "acc_z"] or None to auto-select numeric cols
    threshold = 0.5
    save_csv = "file_predictions.csv"

    predict_file(
        ckpt_path=ckpt_path,
        csv_path=csv_path,
        seq_len=seq_len,
        step_size=step_size,
        cols=cols,
        threshold=threshold,
        save_csv=save_csv,
    )
