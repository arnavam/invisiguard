import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


def evaluate_model_simple(
    model_path="checkpoints/fall-detection-epoch=10-val_loss=0.00.ckpt",
):
    """Simple evaluation for binary classification"""
    from train_fall import FallDataModule, FallDetectionLSTM

    # Load model with explicit map_location to CPU
    model = FallDetectionLSTM.load_from_checkpoint(
        model_path, map_location=torch.device("cpu")
    )
    model.eval()
    model = model.cpu()

    # Setup data
    datamodule = FallDataModule(batch_size=32, seq_len=50)
    datamodule.prepare_data()
    datamodule.setup()

    # Get test dataloader
    test_loader = datamodule.test_dataloader()

    all_preds = []
    all_targets = []
    all_probs = []  # Store probabilities for analysis

    # Run inference
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.cpu()
            outputs = model(x)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long().squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Get unique classes present in data
    unique_classes = np.unique(np.concatenate([all_targets, all_preds]))
    print(f"Unique classes in data: {unique_classes}")
    print(f"Number of unique classes: {len(unique_classes)}")

    # Get class names - handle binary classification properly
    class_names = getattr(datamodule, "class_names", ["Non-Fall", "Fall"])

    # If only one class is present, adjust class names
    if len(unique_classes) == 1:
        print(f"Warning: Only class {unique_classes[0]} is present in predictions")
        # Use the class name for the present class
        if unique_classes[0] < len(class_names):
            class_names = [class_names[unique_classes[0]]]
        else:
            class_names = [f"Class {unique_classes[0]}"]

    # Calculate confusion matrix with explicit labels
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(unique_classes)))

    # Display results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report with explicit labels
    print("\nClassification Report:")
    try:
        # Try with target_names first
        report = classification_report(
            all_targets,
            all_preds,
            target_names=class_names
            if len(class_names) == len(unique_classes)
            else None,
            labels=range(len(unique_classes)),
        )
        print(report)
    except ValueError:
        # If target_names don't match, print without them
        report = classification_report(
            all_targets, all_preds, labels=range(len(unique_classes))
        )
        print(report)

    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Confusion matrix
    if cm.shape[0] == 1:
        # Handle single-class case
        ax1 = axes[0]
        ax1.text(
            0.5,
            0.5,
            f"All predictions: class {unique_classes[0]}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax1.set_title(f"Only Class {unique_classes[0]} Present")
        ax1.axis("off")
    else:
        # Normal confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d", ax=axes[0])
        axes[0].set_title("Confusion Matrix")

    # Plot 2: Prediction distribution
    ax2 = axes[1]
    if len(unique_classes) == 2:
        # For binary classification, show probability distribution
        positive_probs = (
            all_probs[:, 1] if all_probs.shape[1] == 2 else all_probs.flatten()
        )
        ax2.hist(positive_probs[all_targets == 0], alpha=0.5, label="Non-Fall", bins=20)
        ax2.hist(positive_probs[all_targets == 1], alpha=0.5, label="Fall", bins=20)
        ax2.set_xlabel("Predicted Probability of Fall")
        ax2.set_ylabel("Count")
        ax2.set_title("Prediction Probability Distribution")
        ax2.legend()
        ax2.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
        ax2.legend()
    else:
        # For multi-class or single-class, show class distribution
        unique_preds, counts = np.unique(all_preds, return_counts=True)
        ax2.bar([str(c) for c in unique_preds], counts)
        ax2.set_xlabel("Predicted Class")
        ax2.set_ylabel("Count")
        ax2.set_title("Prediction Distribution")

    plt.tight_layout()
    plt.savefig("confusion_matrix_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Calculate detailed metrics
    accuracy = (all_preds == all_targets).mean()

    # For binary classification
    if len(unique_classes) == 2:
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print("\n" + "=" * 50)
        print("DETAILED BINARY CLASSIFICATION METRICS")
        print("=" * 50)
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Create metrics dataframe
        metrics_dict = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"],
            "Value": [accuracy, precision, recall, f1, specificity],
        }
        metrics_df = pd.DataFrame(metrics_dict)
    else:
        # For single class or multi-class
        metrics_df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [accuracy]})

    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))

    # Save predictions for analysis
    results_df = pd.DataFrame(
        {
            "target": all_targets,
            "prediction": all_preds,
            "probability": all_probs[:, 1]
            if all_probs.shape[1] == 2
            else all_probs.flatten(),
        }
    )
    results_df.to_csv("predictions_analysis.csv", index=False)
    print("\nDetailed predictions saved to 'predictions_analysis.csv'")

    return all_preds, all_targets, cm, metrics_df


if __name__ == "__main__":
    # Run evaluation
    evaluate_model_simple()
