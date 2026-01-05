"""
Hyperparameter Tuning for Rule-Based Motion Detector

This script uses your labeled fall/ADL data to find optimal thresholds
for the 4 detection methods:
1. Vector Magnitude Change
2. Jerk Magnitude  
3. Threshold-based Magnitude
4. Rolling Window Statistics

It performs a grid search over threshold combinations and finds
the values that best separate fall (sudden) from ADL (normal) motion.

Author: InvisiGuard Project
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import product
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json


@dataclass
class TuningResult:
    """Results from threshold tuning."""
    thresholds: Dict[str, float]
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def load_raw_data_from_folder(
    data_folder: str = "JO_FALL",
    x_col: str = "Acc_x",
    y_col: str = "Acc_y", 
    z_col: str = "Acc_z",
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load raw accelerometer data from the fall/ADL folder structure.
    
    Returns:
        List of data arrays (each file as one array)
        List of labels (0=ADL/normal, 1=fall/sudden)
    """
    all_data = []
    all_labels = []
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            # Determine label from parent folder
            parent_folder = os.path.basename(os.path.dirname(csv_file)).lower()
            
            if "fall" in parent_folder:
                label = 1  # Fall = sudden motion
            elif "adl" in parent_folder:
                label = 0  # ADL = normal motion
            else:
                continue  # Skip unknown folders
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Find columns (flexible naming)
            possible_x = ['Acc_x', 'acc_x', 'x', 'X', 'accel_x']
            possible_y = ['Acc_y', 'acc_y', 'y', 'Y', 'accel_y']
            possible_z = ['Acc_z', 'acc_z', 'z', 'Z', 'accel_z']
            
            x_col_found = next((c for c in possible_x if c in df.columns), None)
            y_col_found = next((c for c in possible_y if c in df.columns), None)
            z_col_found = next((c for c in possible_z if c in df.columns), None)
            
            if not all([x_col_found, y_col_found, z_col_found]):
                print(f"Skipping {csv_file}: columns not found")
                continue
            
            # Extract data
            data = df[[x_col_found, y_col_found, z_col_found]].values.astype(np.float32)
            
            # Skip files with too little data
            if len(data) < 20:
                continue
            
            # Handle NaN/zero values
            data = np.nan_to_num(data, nan=0.0)
            
            all_data.append(data)
            all_labels.append(label)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    print(f"Loaded {len(all_data)} files: {sum(all_labels)} fall, {len(all_labels) - sum(all_labels)} ADL")
    
    return all_data, all_labels


def compute_features_for_file(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute the 4 method features for a single file.
    
    Returns dict with arrays for each metric over the file.
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    # 1. Magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    # 2. Vector magnitude change / Jerk (same thing)
    jerk_x = np.diff(x, prepend=x[0])
    jerk_y = np.diff(y, prepend=y[0])
    jerk_z = np.diff(z, prepend=z[0])
    jerk = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
    
    # 3. Rolling std (using window of 20)
    window_size = 20
    rolling_std = np.array([
        np.std(magnitude[max(0, i-window_size):i+1]) if i >= window_size else np.std(magnitude[:i+1])
        for i in range(len(magnitude))
    ])
    
    return {
        'magnitude': magnitude,
        'jerk': jerk,
        'rolling_std': rolling_std,
    }


def evaluate_thresholds(
    all_data: List[np.ndarray],
    all_labels: List[int],
    thresholds: Dict[str, float],
    weights: Dict[str, float] = None,
) -> TuningResult:
    """
    Evaluate a set of thresholds on the data.
    
    For each file, we classify it as "sudden" if it has significant sudden motion events.
    """
    if weights is None:
        weights = {
            'vector_magnitude_change': 0.25,
            'magnitude': 0.25,
            'jerk': 0.30,
            'rolling_std': 0.20,
        }
    
    predictions = []
    
    for data in all_data:
        features = compute_features_for_file(data)
        
        # Compute scores for each sample
        scores = []
        for i in range(len(data)):
            mag = features['magnitude'][i]
            jerk = features['jerk'][i]
            std = features['rolling_std'][i]
            
            # Score each method (0-1)
            # Method 1 & 2: Jerk/vector change
            if jerk > thresholds['jerk_sudden']:
                score_jerk = min(1.0, 0.7 + 0.3 * (jerk - thresholds['jerk_sudden']) / thresholds['jerk_sudden'])
            elif jerk < thresholds['jerk_normal']:
                score_jerk = max(0.0, jerk / thresholds['jerk_normal'] * 0.3)
            else:
                score_jerk = 0.3 + 0.4 * (jerk - thresholds['jerk_normal']) / (thresholds['jerk_sudden'] - thresholds['jerk_normal'])
            
            # Method 3: Magnitude threshold
            if mag > thresholds['magnitude_sudden']:
                score_mag = min(1.0, 0.7 + 0.3 * (mag - thresholds['magnitude_sudden']) / thresholds['magnitude_sudden'])
            elif mag < thresholds['magnitude_normal']:
                score_mag = max(0.0, mag / thresholds['magnitude_normal'] * 0.3)
            else:
                score_mag = 0.3 + 0.4 * (mag - thresholds['magnitude_normal']) / (thresholds['magnitude_sudden'] - thresholds['magnitude_normal'])
            
            # Method 4: Rolling std
            if std > thresholds['std_sudden']:
                score_std = min(1.0, 0.7 + 0.3 * (std - thresholds['std_sudden']) / thresholds['std_sudden'])
            elif std < thresholds['std_normal']:
                score_std = max(0.0, std / thresholds['std_normal'] * 0.3)
            else:
                score_std = 0.3 + 0.4 * (std - thresholds['std_normal']) / (thresholds['std_sudden'] - thresholds['std_normal'])
            
            # Combined score
            combined = (
                weights['vector_magnitude_change'] * score_jerk +
                weights['magnitude'] * score_mag +
                weights['jerk'] * score_jerk +
                weights['rolling_std'] * score_std
            )
            scores.append(combined)
        
        scores = np.array(scores)
        
        # Classify file: if a significant portion has high scores, it's a fall
        # Use both max score AND percentage above threshold
        max_score = np.max(scores)
        pct_above_threshold = np.mean(scores > 0.5)
        
        # File is "sudden" if max score > 0.6 OR more than 5% of samples are above 0.5
        if max_score > 0.6 or pct_above_threshold > 0.05:
            predictions.append(1)
        else:
            predictions.append(0)
    
    predictions = np.array(predictions)
    labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix values
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    return TuningResult(
        thresholds=thresholds,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def tune_thresholds(
    data_folder: str = "JO_FALL",
    verbose: bool = True,
) -> Tuple[Dict[str, float], TuningResult]:
    """
    Find optimal thresholds using grid search.
    
    Returns:
        Best thresholds dict and the evaluation result
    """
    print("=" * 60)
    print("HYPERPARAMETER TUNING FOR MOTION DETECTOR")
    print("=" * 60)
    
    # Load data
    all_data, all_labels = load_raw_data_from_folder(data_folder)
    
    if len(all_data) == 0:
        raise ValueError("No data loaded!")
    
    # First, analyze the data to get reasonable search ranges
    print("\nAnalyzing data distributions...")
    
    all_magnitudes = []
    all_jerks = []
    all_stds = []
    
    fall_magnitudes = []
    fall_jerks = []
    adl_magnitudes = []
    adl_jerks = []
    
    for data, label in zip(all_data, all_labels):
        features = compute_features_for_file(data)
        all_magnitudes.extend(features['magnitude'])
        all_jerks.extend(features['jerk'])
        all_stds.extend(features['rolling_std'])
        
        if label == 1:  # Fall
            fall_magnitudes.extend(features['magnitude'])
            fall_jerks.extend(features['jerk'])
        else:  # ADL
            adl_magnitudes.extend(features['magnitude'])
            adl_jerks.extend(features['jerk'])
    
    # Print statistics
    print(f"\nMagnitude stats:")
    print(f"  All data: mean={np.mean(all_magnitudes):.3f}, max={np.max(all_magnitudes):.3f}")
    print(f"  Fall: mean={np.mean(fall_magnitudes):.3f}, max={np.max(fall_magnitudes):.3f}")
    print(f"  ADL:  mean={np.mean(adl_magnitudes):.3f}, max={np.max(adl_magnitudes):.3f}")
    
    print(f"\nJerk stats:")
    print(f"  All data: mean={np.mean(all_jerks):.3f}, max={np.max(all_jerks):.3f}")
    print(f"  Fall: mean={np.mean(fall_jerks):.3f}, max={np.max(fall_jerks):.3f}")
    print(f"  ADL:  mean={np.mean(adl_jerks):.3f}, max={np.max(adl_jerks):.3f}")
    
    print(f"\nRolling Std stats:")
    print(f"  All data: mean={np.mean(all_stds):.3f}, max={np.max(all_stds):.3f}")
    
    # Define search space based on data analysis
    # Use percentiles to set reasonable ranges
    mag_p50 = np.percentile(all_magnitudes, 50)
    mag_p90 = np.percentile(all_magnitudes, 90)
    mag_p99 = np.percentile(all_magnitudes, 99)
    
    jerk_p50 = np.percentile(all_jerks, 50)
    jerk_p90 = np.percentile(all_jerks, 90)
    jerk_p99 = np.percentile(all_jerks, 99)
    
    std_p50 = np.percentile(all_stds, 50)
    std_p90 = np.percentile(all_stds, 90)
    
    print(f"\nPercentiles:")
    print(f"  Magnitude: p50={mag_p50:.3f}, p90={mag_p90:.3f}, p99={mag_p99:.3f}")
    print(f"  Jerk: p50={jerk_p50:.3f}, p90={jerk_p90:.3f}, p99={jerk_p99:.3f}")
    print(f"  Std: p50={std_p50:.3f}, p90={std_p90:.3f}")
    
    # Grid search ranges
    magnitude_normal_range = np.linspace(mag_p50 * 0.5, mag_p90, 5)
    magnitude_sudden_range = np.linspace(mag_p90, mag_p99 * 1.2, 5)
    jerk_normal_range = np.linspace(jerk_p50 * 0.5, jerk_p90, 5)
    jerk_sudden_range = np.linspace(jerk_p90, jerk_p99 * 1.2, 5)
    std_normal_range = np.linspace(std_p50 * 0.3, std_p50, 3)
    std_sudden_range = np.linspace(std_p50, std_p90 * 1.5, 3)
    
    print(f"\nSearch space size: {len(magnitude_normal_range) * len(magnitude_sudden_range) * len(jerk_normal_range) * len(jerk_sudden_range) * len(std_normal_range) * len(std_sudden_range)} combinations")
    print("Starting grid search...")
    
    best_result = None
    best_f1 = 0
    
    total_combinations = (len(magnitude_normal_range) * len(magnitude_sudden_range) * 
                          len(jerk_normal_range) * len(jerk_sudden_range) *
                          len(std_normal_range) * len(std_sudden_range))
    
    tested = 0
    for mag_normal in magnitude_normal_range:
        for mag_sudden in magnitude_sudden_range:
            if mag_sudden <= mag_normal:
                continue
            for jerk_normal in jerk_normal_range:
                for jerk_sudden in jerk_sudden_range:
                    if jerk_sudden <= jerk_normal:
                        continue
                    for std_normal in std_normal_range:
                        for std_sudden in std_sudden_range:
                            if std_sudden <= std_normal:
                                continue
                            
                            thresholds = {
                                'magnitude_normal': mag_normal,
                                'magnitude_sudden': mag_sudden,
                                'jerk_normal': jerk_normal,
                                'jerk_sudden': jerk_sudden,
                                'std_normal': std_normal,
                                'std_sudden': std_sudden,
                                'stationary_threshold': 0.1,
                            }
                            
                            result = evaluate_thresholds(all_data, all_labels, thresholds)
                            tested += 1
                            
                            if result.f1 > best_f1:
                                best_f1 = result.f1
                                best_result = result
                                if verbose:
                                    print(f"  New best F1: {best_f1:.4f} (tested {tested})")
    
    print(f"\nTested {tested} valid threshold combinations")
    
    if best_result is None:
        print("No valid result found!")
        return None, None
    
    print("\n" + "=" * 60)
    print("BEST THRESHOLDS FOUND:")
    print("=" * 60)
    print(f"  magnitude_normal: {best_result.thresholds['magnitude_normal']:.4f}")
    print(f"  magnitude_sudden: {best_result.thresholds['magnitude_sudden']:.4f}")
    print(f"  jerk_normal: {best_result.thresholds['jerk_normal']:.4f}")
    print(f"  jerk_sudden: {best_result.thresholds['jerk_sudden']:.4f}")
    print(f"  std_normal: {best_result.thresholds['std_normal']:.4f}")
    print(f"  std_sudden: {best_result.thresholds['std_sudden']:.4f}")
    
    print(f"\nPERFORMANCE:")
    print(f"  Accuracy:  {best_result.accuracy:.4f}")
    print(f"  Precision: {best_result.precision:.4f}")
    print(f"  Recall:    {best_result.recall:.4f}")
    print(f"  F1 Score:  {best_result.f1:.4f}")
    print(f"  False Positives: {best_result.false_positives}")
    print(f"  False Negatives: {best_result.false_negatives}")
    
    # Save best thresholds to file
    output_file = "optimized_thresholds.json"
    with open(output_file, 'w') as f:
        json.dump(best_result.thresholds, f, indent=2)
    print(f"\nThresholds saved to: {output_file}")
    
    # Print code snippet to use
    print("\n" + "=" * 60)
    print("TO USE OPTIMIZED THRESHOLDS:")
    print("=" * 60)
    print("""
from detect_motion_rulebased import SuddenMotionDetector
import json

# Load optimized thresholds
with open('optimized_thresholds.json', 'r') as f:
    thresholds = json.load(f)

# Create detector with optimized thresholds
detector = SuddenMotionDetector(thresholds=thresholds)
""")
    
    return best_result.thresholds, best_result


if __name__ == "__main__":
    # Run tuning
    best_thresholds, best_result = tune_thresholds(
        data_folder="JO_FALL",
        verbose=True,
    )
