from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from create_dataset import create_fall_adl_dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, random_split

# 1. Custom Dataset
class FallDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, 2) containing (x, y) sequences
            labels: Array of shape (n_samples,) containing binary labels (0: no fall, 1: fall)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# 2. Lightning Data Module
class FallDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = None,
        sequences: np.ndarray = None,
        labels: np.ndarray = None,
        batch_size: int = 32,
        seq_len: int = 50,
        train_split: float = 0.7,
        val_split: float = 0.15,
    ):
        super().__init__()
        self.data_path = data_path
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_split = train_split
        self.val_split = val_split

    def prepare_data(self):
        """Load and prepare data if using file path"""
        sequences, labels, label_map, feature_names = create_fall_adl_dataset(
            data_folder="JO_FALL",
            seq_len=100,  # Adjust based on your data
            step_size=20,
            normalize=True,
        )
        rng = np.random.RandomState(42)  # For reproducibility, optional
        idx = rng.permutation(len(sequences))
        sequences = sequences[idx]
        labels = labels[idx]
        # label_map = label_map[idx] if isinstance(label_map, np.ndarray) else np.array(label_map)[idx]
        # selected_columns = selected_columns[idx] if isinstance(selected_columns, np.ndarray) else np.array(selected_columns)[idx]

        self.sequences = sequences
        self.labels = labels

    def setup(self, stage: Optional[str] = None):
        """Split data into train, val, test"""
        if self.sequences is None or self.labels is None:
            # Generate synthetic data for demonstration
            n_samples = 1000
            self.sequences = np.random.randn(n_samples, self.seq_len, 2)
            self.labels = np.random.randint(0, 2, n_samples)

        dataset = FallDataset(self.sequences, self.labels)

        # Calculate split sizes
        n_total = len(dataset)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        n_test = n_total - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


# 3. Lightning LSTM Model
class FallDetectionLSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim



        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Option 1: Use last hidden state
        if self.hparams.bidirectional:
            # For bidirectional, concatenate last forward and backward hidden states
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        # Option 2: Use attention over sequence (uncomment to use)
        # attention_weights = self.attention(lstm_out)
        # context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # logits = self.classifier(context_vector)

        # Using last hidden state
        logits = self.classifier(last_hidden)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)
        loss = self.criterion(logits, labels.float())

        preds = torch.sigmoid(logits) > 0.5
        preds = preds.long()

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # Store for epoch metrics
        self.training_step_outputs.append(
            {"loss": loss, "preds": preds, "labels": labels}
        )

        return loss

    def on_train_epoch_end(self):
        # Calculate epoch metrics
        outputs = self.training_step_outputs

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        all_preds = torch.cat([x["preds"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])

        epoch_acc = (all_preds == all_labels).float().mean()

        self.log("train_epoch_loss", avg_loss)
        self.log("train_epoch_acc", epoch_acc)

        # Clear memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        self.validation_step_outputs.append(
            {"loss": loss, "preds": preds, "labels": labels}
        )

        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        if outputs:
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            all_preds = torch.cat([x["preds"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])

            epoch_acc = (all_preds == all_labels).float().mean()

            self.log("val_epoch_loss", avg_loss)
            self.log("val_epoch_acc", epoch_acc)

            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)

        self.test_step_outputs.append({"loss": loss, "preds": preds, "labels": labels})

        return loss

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        if outputs:
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            all_preds = torch.cat([x["preds"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])

            # Calculate comprehensive metrics
            test_acc = (all_preds == all_labels).float().mean()

            # Convert to numpy for sklearn metrics
            preds_np = all_preds.cpu().numpy()
            labels_np = all_labels.cpu().numpy()

            precision = precision_score(labels_np, preds_np, zero_division=0)
            recall = recall_score(labels_np, preds_np, zero_division=0)
            f1 = f1_score(labels_np, preds_np, zero_division=0)

            print("\nTest Results:")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Loss: {avg_loss:.4f}")

            # Confusion matrix
            cm = confusion_matrix(labels_np, preds_np)
            print(f"Confusion Matrix:\n{cm}")

            self.test_step_outputs.clear()


# 4. Main training function
def train_model():
    """Train the fall detection model"""

    # Initialize data module
    data_module = FallDataModule(
        batch_size=32,
        seq_len=50,
    )

    # Initialize model
    model = FallDetectionLSTM(
        input_dim=3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        bidirectional=True,
    )

    # Print an accurate FP32 model summary (even if training will use mixed precision)
    print(ModelSummary(model.float(), max_depth=2))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="fall-detection-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
    )

    # Logger
    logger = TensorBoardLogger("logs/", name="fall_detection")

    # Trainer: use mixed precision on GPU, but disable Lightning's built-in summary
    use_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator="gpu" if use_cuda else "cpu",
        devices=1 if use_cuda else None,
        log_every_n_steps=10,
        precision=16 if use_cuda else 32,  # train in 16-bit on GPU
        logger=logger,
        enable_model_summary=False,        # we've printed our own FP32 summary
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    return model, trainer, data_module


# 5. Inference function
def predict_fall(model: FallDetectionLSTM, sequence: np.ndarray) -> Tuple[int, float]:
    """
    Predict if a sequence contains a fall

    Args:
        model: Trained model
        sequence: Array of shape (seq_len, 2) containing (x, y) coordinates

    Returns:
        prediction (0 or 1), confidence score
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

        # Move to same device as model
        device = next(model.parameters()).device
        sequence_tensor = sequence_tensor.to(device)

        # Get prediction
        logit = model(sequence_tensor)
        probability = torch.sigmoid(logit).item()
        prediction = 1 if probability > 0.5 else 0

        return prediction, probability


# 6. Example usage
if __name__ == "__main__":
    # Train the model and get the data module back
    model, trainer, data_module = train_model()

    # Use a real sample from the test dataset instead of a random sequence
    # data_module.test_dataset is a Subset; indexing returns (sequence_tensor, label_tensor)
    sample_seq_tensor, sample_label_tensor = data_module.test_dataset[0]
    sample_seq = sample_seq_tensor.cpu().numpy() if isinstance(sample_seq_tensor, torch.Tensor) else sample_seq_tensor
    sample_label = int(sample_label_tensor.item()) if isinstance(sample_label_tensor, torch.Tensor) else int(sample_label_tensor)

    prediction, confidence = predict_fall(model, sample_seq)

    print(f"\nGround truth: {'Fall' if sample_label == 1 else 'No Fall'}")
    print(f"Prediction: {'Fall' if prediction == 1 else 'No Fall'}")
    print(f"Confidence: {confidence:.2%}")
