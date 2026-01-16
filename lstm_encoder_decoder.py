"""
LSTM Encoder-Decoder for Event Detection (e.g., Clapping Detection)

This module implements an LSTM-based autoencoder that can detect shifts in events
by learning normal patterns and identifying anomalies when events (like clapping) occur.
The encoder compresses the sequence, and the decoder reconstructs it.
High reconstruction error indicates an event/shift.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from create_dataset import create_fall_adl_dataset, create_sliding_windows_multi
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, random_split


# ============================================================================
# 1. Custom Dataset for Sequence Data
# ============================================================================
class SequenceDataset(Dataset):
    """Dataset for sequence data with optional labels."""

    def __init__(
        self, sequences: np.ndarray, labels: Optional[np.ndarray] = None
    ):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
            labels: Optional array of shape (n_samples,) for supervised mode
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx], self.sequences[idx]  # For autoencoder


# ============================================================================
# 2. LSTM Encoder Module
# ============================================================================
class LSTMEncoder(nn.Module):
    """
    LSTM Encoder that compresses input sequences into a latent representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Dimension of the latent representation
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Project to latent space
        self.fc_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input sequence to latent representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            latent: Latent representation (batch, latent_dim)
            hidden: Tuple of (h_n, c_n) for decoder initialization
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last output from both directions
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            last_output = torch.cat(
                [lstm_out[:, -1, : self.hidden_dim], lstm_out[:, 0, self.hidden_dim :]],
                dim=1,
            )
        else:
            last_output = lstm_out[:, -1, :]

        # Project to latent space
        latent = self.fc_latent(last_output)
        latent = self.layer_norm(latent)

        return latent, (h_n, c_n)


# ============================================================================
# 3. LSTM Decoder Module
# ============================================================================
class LSTMDecoder(nn.Module):
    """
    LSTM Decoder that reconstructs sequences from latent representation.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.3,
    ):
        """
        Args:
            output_dim: Number of output features per timestep
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Dimension of the latent representation
            num_layers: Number of LSTM layers
            seq_len: Length of output sequence
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_dim = output_dim

        # Project latent to hidden state
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to sequence.

        Args:
            latent: Latent tensor of shape (batch, latent_dim)

        Returns:
            output: Reconstructed sequence (batch, seq_len, output_dim)
        """
        batch_size = latent.size(0)

        # Initialize hidden state from latent
        h_0 = self.fc_hidden(latent)
        h_0 = h_0.view(batch_size, self.num_layers, self.hidden_dim)
        h_0 = h_0.permute(1, 0, 2).contiguous()

        c_0 = self.fc_cell(latent)
        c_0 = c_0.view(batch_size, self.num_layers, self.hidden_dim)
        c_0 = c_0.permute(1, 0, 2).contiguous()

        # Repeat latent across sequence length as input
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(decoder_input, (h_0, c_0))

        # Project to output dimension
        output = self.fc_out(lstm_out)

        return output


# ============================================================================
# 4. LSTM Autoencoder (Encoder + Decoder)
# ============================================================================
class LSTMAutoencoder(pl.LightningModule):
    """
    Complete LSTM Autoencoder for event/shift detection.

    The model learns to reconstruct normal patterns. During inference,
    high reconstruction error indicates an event (like clapping) or shift.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        reconstruction_threshold: float = 0.5,
        bidirectional_encoder: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Dimension of latent representation
            num_layers: Number of LSTM layers
            seq_len: Sequence length
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            reconstruction_threshold: Threshold for anomaly detection
            bidirectional_encoder: Use bidirectional encoder
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.reconstruction_threshold = reconstruction_threshold
        self.seq_len = seq_len

        # Encoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )

        # Decoder
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
        )

        # Loss function
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        # For tracking metrics
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            reconstructed: Reconstructed sequence
            latent: Latent representation
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation only."""
        latent, _ = self.encoder(x)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation."""
        return self.decoder(latent)

    def compute_reconstruction_error(
        self, x: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction error.

        Args:
            x: Original input
            reconstructed: Reconstructed output

        Returns:
            error: Per-sample reconstruction error (batch,)
        """
        # MSE per sample, averaged over sequence and features
        error = self.reconstruction_loss(reconstructed, x)
        error = error.mean(dim=[1, 2])  # Average over seq_len and features
        return error

    def detect_event(
        self, x: torch.Tensor, threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect events/shifts based on reconstruction error.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            threshold: Optional threshold (uses self.reconstruction_threshold if None)

        Returns:
            is_event: Boolean tensor indicating event detection
            error: Reconstruction error scores
        """
        threshold = threshold or self.reconstruction_threshold
        reconstructed, _ = self.forward(x)
        error = self.compute_reconstruction_error(x, reconstructed)
        is_event = error > threshold
        return is_event, error

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _shared_step(self, batch, batch_idx) -> torch.Tensor:
        """Shared step for training and validation."""
        x, _ = batch  # Ignore labels for autoencoder training
        reconstructed, _ = self.forward(x)
        loss = self.reconstruction_loss(reconstructed, x).mean()
        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self._shared_step(batch, batch_idx)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Log average training loss."""
        if self.train_losses:
            avg_loss = np.mean(self.train_losses)
            self.log("avg_train_loss", avg_loss)
            self.train_losses = []

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self._shared_step(batch, batch_idx)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Log average validation loss."""
        if self.val_losses:
            avg_loss = np.mean(self.val_losses)
            self.log("avg_val_loss", avg_loss)
            self.val_losses = []

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss


# ============================================================================
# 5. Supervised Event Classifier using Encoder
# ============================================================================
class LSTMEventClassifier(pl.LightningModule):
    """
    LSTM-based classifier for event detection (e.g., clapping vs normal).

    Uses the encoder architecture with a classification head.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension of LSTM
            latent_dim: Dimension of latent representation
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            learning_rate: Learning rate
            class_weights: Optional weights for imbalanced classes
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Encoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics tracking
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            logits: Classification logits (batch, num_classes)
        """
        latent, _ = self.encoder(x)
        logits = self.classifier(latent)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        latent, _ = self.encoder(x)
        return latent

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class and confidence.

        Returns:
            predictions: Predicted class labels
            confidences: Prediction confidences (probabilities)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
            },
        }

    def _shared_step(self, batch, batch_idx):
        """Shared step for training and validation."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.train_preds.extend(preds.cpu().numpy())
        self.train_labels.extend(y.cpu().numpy())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Compute and log training metrics."""
        if self.train_preds:
            acc = accuracy_score(self.train_labels, self.train_preds)
            f1 = f1_score(self.train_labels, self.train_preds, average="weighted")
            self.log("train_acc", acc)
            self.log("train_f1", f1)
            self.train_preds = []
            self.train_labels = []

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        if self.val_preds:
            acc = accuracy_score(self.val_labels, self.val_preds)
            f1 = f1_score(self.val_labels, self.val_preds, average="weighted")
            precision = precision_score(
                self.val_labels, self.val_preds, average="weighted", zero_division=0
            )
            recall = recall_score(
                self.val_labels, self.val_preds, average="weighted", zero_division=0
            )
            self.log("val_acc", acc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.val_preds = []
            self.val_labels = []

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds, y = self._shared_step(batch, batch_idx)
        self.test_preds.extend(preds.cpu().numpy())
        self.test_labels.extend(y.cpu().numpy())
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        if self.test_preds:
            acc = accuracy_score(self.test_labels, self.test_preds)
            f1 = f1_score(self.test_labels, self.test_preds, average="weighted")
            precision = precision_score(
                self.test_labels, self.test_preds, average="weighted", zero_division=0
            )
            recall = recall_score(
                self.test_labels, self.test_preds, average="weighted", zero_division=0
            )
            cm = confusion_matrix(self.test_labels, self.test_preds)

            print("\n" + "=" * 50)
            print("Test Results:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
            print("=" * 50)

            self.log("test_acc", acc)
            self.log("test_f1", f1)
            self.log("test_precision", precision)
            self.log("test_recall", recall)


# ============================================================================
# 6. Data Module for Training
# ============================================================================
class EventDataModule(pl.LightningDataModule):
    """Data module for event detection datasets."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        num_workers: int = 0,
    ):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
            labels: Array of shape (n_samples,)
            batch_size: Batch size
            train_split: Fraction for training
            val_split: Fraction for validation
            num_workers: Number of data loading workers
        """
        super().__init__()
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test splits."""
        dataset = SequenceDataset(self.sequences, self.labels)

        n_total = len(dataset)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        n_test = n_total - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Dataset splits: Train={n_train}, Val={n_val}, Test={n_test}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ============================================================================
# 7. Training Functions
# ============================================================================
def train_autoencoder(
    sequences: np.ndarray,
    labels: Optional[np.ndarray] = None,
    input_dim: int = 3,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 2,
    seq_len: int = 50,
    batch_size: int = 32,
    max_epochs: int = 50,
    learning_rate: float = 0.001,
) -> LSTMAutoencoder:
    """
    Train the LSTM autoencoder for event detection.

    Args:
        sequences: Training sequences (n_samples, seq_len, n_features)
        labels: Optional labels (used for splitting only)
        Other args: Model hyperparameters

    Returns:
        Trained model
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    # Create data module
    if labels is None:
        labels = np.zeros(len(sequences), dtype=np.int64)

    data_module = EventDataModule(
        sequences=sequences, labels=labels, batch_size=batch_size
    )

    # Create model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        learning_rate=learning_rate,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            dirpath="checkpoints/autoencoder",
            filename="lstm_autoencoder-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module)

    return model


def train_classifier(
    sequences: np.ndarray,
    labels: np.ndarray,
    input_dim: int = 3,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 2,
    num_classes: int = 2,
    batch_size: int = 32,
    max_epochs: int = 50,
    learning_rate: float = 0.001,
) -> LSTMEventClassifier:
    """
    Train the LSTM classifier for event detection.

    Args:
        sequences: Training sequences (n_samples, seq_len, n_features)
        labels: Labels for each sequence
        Other args: Model hyperparameters

    Returns:
        Trained model
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    # Compute class weights for imbalanced data
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = torch.FloatTensor(len(labels) / (len(unique_labels) * counts))

    # Create data module
    data_module = EventDataModule(
        sequences=sequences, labels=labels, batch_size=batch_size
    )

    # Create model
    model = LSTMEventClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        learning_rate=learning_rate,
        class_weights=class_weights,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_f1", patience=10, mode="max"),
        ModelCheckpoint(
            dirpath="checkpoints/classifier",
            filename="lstm_classifier-{epoch:02d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    return model


# ============================================================================
# 8. Inference Functions
# ============================================================================
def detect_shift_with_autoencoder(
    model: LSTMAutoencoder, sequence: np.ndarray, threshold: Optional[float] = None
) -> Tuple[bool, float]:
    """
    Detect event/shift in a single sequence using autoencoder.

    Args:
        model: Trained autoencoder
        sequence: Sequence of shape (seq_len, n_features)
        threshold: Detection threshold (uses model's threshold if None)

    Returns:
        is_event: Whether an event was detected
        error: Reconstruction error score
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        is_event, error = model.detect_event(x, threshold)
        return bool(is_event.item()), float(error.item())


def classify_event(
    model: LSTMEventClassifier, sequence: np.ndarray
) -> Tuple[int, float]:
    """
    Classify a single sequence.

    Args:
        model: Trained classifier
        sequence: Sequence of shape (seq_len, n_features)

    Returns:
        prediction: Predicted class
        confidence: Prediction confidence
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        preds, confs = model.predict(x)
        return int(preds.item()), float(confs.item())


def compute_threshold_from_normal(
    model: LSTMAutoencoder, normal_sequences: np.ndarray, percentile: float = 95
) -> float:
    """
    Compute anomaly detection threshold from normal data.

    Args:
        model: Trained autoencoder
        normal_sequences: Sequences of normal (non-event) data
        percentile: Percentile for threshold (e.g., 95 means 95% of normal data is below threshold)

    Returns:
        threshold: Computed threshold value
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(normal_sequences), 32):
            batch = torch.FloatTensor(normal_sequences[i : i + 32])
            reconstructed, _ = model(batch)
            error = model.compute_reconstruction_error(batch, reconstructed)
            errors.extend(error.cpu().numpy())

    threshold = np.percentile(errors, percentile)
    print(f"Computed threshold at {percentile}th percentile: {threshold:.6f}")
    return threshold


# ============================================================================
# 9. Main Example Usage
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM Encoder-Decoder for Event Detection")
    print("=" * 60)

    # Load data using create_dataset
    data_folder = "JO_FALL"

    print("\n1. Loading dataset...")
    sequences, labels, label_map, feature_names = create_fall_adl_dataset(
        data_folder=data_folder,
        seq_len=100,
        step_size=20,
        normalize=True,
    )

    print(f"\nDataset loaded:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label mapping: {label_map}")
    print(f"  Features: {feature_names}")

    input_dim = sequences.shape[2]
    seq_len = sequences.shape[1]

    # Option 1: Train Autoencoder for anomaly/shift detection
    print("\n2. Training LSTM Autoencoder...")
    print("-" * 40)

    autoencoder = train_autoencoder(
        sequences=sequences,
        labels=labels,
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        seq_len=seq_len,
        batch_size=32,
        max_epochs=30,
    )

    # Compute threshold from normal (ADL) data
    normal_mask = labels == label_map.get("adl", 0)
    if np.any(normal_mask):
        normal_sequences = sequences[normal_mask]
        threshold = compute_threshold_from_normal(autoencoder, normal_sequences)
        autoencoder.reconstruction_threshold = threshold

    # Option 2: Train Classifier for supervised event detection
    print("\n3. Training LSTM Classifier...")
    print("-" * 40)

    classifier = train_classifier(
        sequences=sequences,
        labels=labels,
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        num_classes=len(label_map),
        batch_size=32,
        max_epochs=30,
    )

    # Example inference
    print("\n4. Example Inference...")
    print("-" * 40)

    test_sequence = sequences[0]  # Take first sequence as example

    # Autoencoder-based detection
    is_event, error = detect_shift_with_autoencoder(autoencoder, test_sequence)
    print(f"Autoencoder - Event detected: {is_event}, Error: {error:.6f}")

    # Classifier-based detection
    pred_class, confidence = classify_event(classifier, test_sequence)
    label_to_name = {v: k for k, v in label_map.items()}
    print(
        f"Classifier - Predicted: {label_to_name[pred_class]}, Confidence: {confidence:.4f}"
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
