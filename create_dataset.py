import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def create_dataset_from_folder_structure(
    data_folder: str,
    seq_len: int = 50,
    step_size: int = 10,
    label_map: Optional[Dict[str, int]] = None,
    normalize: bool = True,
    use_columns: Optional[List[Union[str, int]]] = None,
    exclude_columns: Optional[List[str]] = None,
    recursive: bool = True,
    label_from_parent: bool = True,  # Get label from parent folder
    label_folder_names: Optional[
        List[str]
    ] = None,  # Specific folders to consider as labels
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    """
    Create dataset from nested folder structure where labels come from parent folders.

    Args:
        data_folder: Root path containing the data
        seq_len: Length of sequences to create
        step_size: Step size for sliding window
        label_map: Optional mapping from folder name to label
        normalize: Whether to normalize the data per feature
        use_columns: Specific columns to use
        exclude_columns: Columns to exclude
        recursive: Whether to search for CSV files recursively
        label_from_parent: If True, use parent folder name as label
        label_folder_names: Specific folder names to use as labels (e.g., ['fall', 'adl'])
                           If None, uses immediate parent folder of CSV files

    Returns:
        sequences: Array of shape (n_samples, seq_len, n_features)
        labels: Array of shape (n_samples,)
        label_mapping: Dictionary mapping labels to indices
        feature_names: List of feature/column names used
    """

    all_sequences = []
    all_labels = []

    # Find all CSV files
    if recursive:
        csv_files = glob.glob(os.path.join(data_folder, "**", "*.csv"), recursive=True)
    else:
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_folder}")

    print(f"Found {len(csv_files)} CSV files")

    # Determine feature names from first file
    first_file = csv_files[0]
    first_df = pd.read_csv(first_file)

    # Determine which columns to use
    if use_columns is not None:
        if all(isinstance(col, int) for col in use_columns):
            selected_columns = [
                first_df.columns[i] for i in use_columns if i < len(first_df.columns)
            ]
        else:
            selected_columns = [col for col in use_columns if col in first_df.columns]
    else:
        selected_columns = first_df.select_dtypes(include=[np.number]).columns.tolist()

    if exclude_columns:
        selected_columns = [
            col for col in selected_columns if col not in exclude_columns
        ]

    if not selected_columns:
        raise ValueError("No numeric columns found or selected for processing")

    print(f"Using columns: {selected_columns}")
    print(f"Number of features: {len(selected_columns)}")

    # Collect all unique labels from folder structure
    folder_labels = set()
    for csv_file in csv_files:
        if label_from_parent:
            # Get immediate parent folder name
            parent_folder = Path(csv_file).parent.name
            folder_labels.add(parent_folder)
        else:
            # Fallback: use filename stem
            filename = Path(csv_file).stem
            folder_labels.add(filename)

    # Filter labels if specific folder names are provided
    if label_folder_names is not None:
        folder_labels = {
            label for label in folder_labels if label in label_folder_names
        }

    # Create label mapping if not provided
    if label_map is None:
        unique_labels = sorted(folder_labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

    print(f"Label mapping: {label_map}")

    # Statistics for normalization
    if normalize:
        all_features_data = []

    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Determine label based on parent folder
            if label_from_parent:
                label_name = Path(csv_file).parent.name
                # Skip if label folder name is not in our label map (unless we're using all)
                if (
                    label_folder_names is not None
                    and label_name not in label_folder_names
                ):
                    continue
            else:
                label_name = Path(csv_file).stem

            label = label_map.get(label_name)

            if label is None:
                # Create new label if not in map and we're not using specific folders
                if label_folder_names is None:
                    new_idx = len(label_map)
                    label_map[label_name] = new_idx
                    label = new_idx
                    print(f"Added new label: {label_name} -> {new_idx}")
                else:
                    continue

            # Read CSV file
            df = pd.read_csv(csv_file)

            # Select columns
            available_columns = [col for col in selected_columns if col in df.columns]

            if not available_columns:
                print(
                    f"Warning: None of the selected columns found in {
                        csv_file
                    }, skipping"
                )
                continue

            # Extract values for selected columns
            feature_values = []
            missing_columns = []
            for col in selected_columns:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce").values.astype(
                        np.float32
                    )
                    # Handle NaN values
                    if np.isnan(values).any():
                        col_mean = np.nanmean(values)
                        values = np.where(np.isnan(values), col_mean, values)
                    feature_values.append(values)
                else:
                    # Track missing columns but don't add zeros
                    missing_columns.append(col)

            if missing_columns:
                print(
                    f"Warning: Columns {missing_columns} not found in {
                        csv_file
                    }, skipping file"
                )
                continue

            # Stack features
            data = np.column_stack(feature_values)

            # Collect for normalization if needed
            if normalize:
                all_features_data.append(data)

            # Create sequences
            sequences = create_sliding_windows_multi(data, seq_len, step_size)

            # Add sequences and labels
            all_sequences.extend(sequences)
            all_labels.extend([label] * len(sequences))

            print(
                f"Processed {csv_file}: {len(sequences)} sequences, label: {
                    label_name
                } ({label})"
            )

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Convert to numpy arrays
    if not all_sequences:
        raise ValueError("No valid sequences created from the data files")

    sequences_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    # Normalize if requested
    if normalize and all_features_data:
        sequences_array = normalize_sequences_per_feature(sequences_array)

    # Print dataset statistics
    print("\n" + "=" * 50)
    print("Dataset created successfully:")
    print(f"  Total sequences: {len(sequences_array)}")
    print(f"  Sequence shape: {sequences_array.shape}")
    print(f"  Number of features: {sequences_array.shape[2]}")
    print("  Label distribution:")

    # Reverse label mapping for printing
    label_to_name = {v: k for k, v in label_map.items()}
    for label_idx in sorted(label_to_name.keys()):
        label_name = label_to_name[label_idx]
        count = np.sum(labels_array == label_idx)
        if len(labels_array) > 0:
            print(
                f"    {label_name}: {count} sequences ({
                    count / len(labels_array) * 100:.1f}%)"
            )
        else:
            print(f"    {label_name}: {count} sequences")

    return sequences_array, labels_array, label_map, selected_columns


def create_sliding_windows_multi(
    data: np.ndarray, seq_len: int, step_size: int
) -> List[np.ndarray]:
    """
    Create sliding windows for multi-dimensional data.
    """
    sequences = []
    n_samples = data.shape[0]

    if n_samples < seq_len:
        # If file has fewer samples than seq_len, pad with zeros
        padded = np.zeros((seq_len, data.shape[1]), dtype=np.float32)
        padded[:n_samples] = data
        sequences.append(padded)
        return sequences

    for start_idx in range(0, n_samples - seq_len + 1, step_size):
        end_idx = start_idx + seq_len
        sequence = data[start_idx:end_idx]
        sequences.append(sequence)

    return sequences


def normalize_sequences_per_feature(sequences: np.ndarray) -> np.ndarray:
    """
    Normalize each feature independently using z-score normalization.
    """
    n_features = sequences.shape[2]
    normalized = sequences.copy()

    for feature_idx in range(n_features):
        feature_data = sequences[:, :, feature_idx]
        mean = np.mean(feature_data)
        std = np.std(feature_data)

        if std > 0:
            normalized[:, :, feature_idx] = (feature_data - mean) / std
        else:
            normalized[:, :, feature_idx] = feature_data - mean

    return normalized


# Simplified version specifically for your folder structure
def create_fall_adl_dataset(
    data_folder: str,
    seq_len: int = 50,
    step_size: int = 10,
    normalize: bool = True,
    label_map: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    """
    Specialized function for fall/ADL dataset structure.

    Args:
        data_folder: Root folder (e.g., 'JO_FALL')
        seq_len: Length of sequences
        step_size: Step size for sliding window
        normalize: Whether to normalize data
        label_map: Optional label mapping (default: {'adl': 0, 'fall': 1})
        **kwargs: Additional arguments passed to create_dataset_from_folder_structure

    Returns:
        sequences, labels, label_map, feature_names
    """

    if label_map is None:
        # Default mapping for fall detection
        label_map = {"adl": 0, "fall": 1}

    print("Creating fall/ADL dataset...")
    print(f"Looking for CSV files in: {data_folder}")

    # Use the folder structure approach
    return create_dataset_from_folder_structure(
        data_folder=data_folder,
        seq_len=seq_len,
        step_size=step_size,
        normalize=normalize,
        label_map=label_map,
        label_from_parent=True,
        # Only use these folder names as labels
        label_folder_names=["adl", "fall"],
        recursive=True,
        **kwargs,
    )


# Helper function to print folder structure for debugging
def explore_folder_structure(root_folder: str, max_depth: int = 4):
    """Print folder structure to understand data organization."""
    print(f"Exploring folder structure of: {root_folder}")
    print("-" * 50)

    for root, dirs, files in os.walk(root_folder):
        level = root.replace(root_folder, "").count(os.sep)
        if level <= max_depth:
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            if level == max_depth - 1:
                # Show CSV files at this level
                csv_files = [f for f in files if f.endswith(".csv")]
                for f in csv_files[:5]:  # Show first 5
                    print(f"{subindent}{f}")
                if len(csv_files) > 5:
                    print(f"{subindent}... and {len(csv_files) - 5} more")
    print("-" * 50)


# Example usage
if __name__ == "__main__":
    # First, explore the folder structure
    data_root = "JO_FALL"
    explore_folder_structure(data_root)

    # Create dataset using the specialized function
    sequences, labels, label_map, feature_names = create_fall_adl_dataset(
        data_folder=data_root,
        seq_len=100,  # Adjust based on your data
        step_size=20,
        normalize=True,
    )

    print(f"\nDataset shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label mapping: {label_map}")
    print(f"Features used: {feature_names}")

    # # You can also use the general function with custom settings
    # sequences2, labels2, label_map2, feature_names2 = (
    #     create_dataset_from_folder_structure(
    #         data_folder=data_root,
    #         seq_len=100,
    #         step_size=20,
    #         normalize=True,
    #         label_map={"adl": 0, "fall": 1},
    #         label_from_parent=True,
    #         label_folder_names=["adl", "fall"],
    #     )
    # )
