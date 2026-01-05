"""
ML-Based Motion Detection: Sudden vs Normal Fluctuation Classifier

This module provides an LSTM-based ML model to classify motion patterns
as "sudden" (large fluctuations) or "normal" (walking/gradual movement).

Uses the same architecture as fall detection but trained for motion classification.
Can be trained on your own labeled data or used with the rule-based detector
to generate training labels automatically.

Author: InvisiGuard Project
"""

from typing import Optional, Tuple, List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from torch.utils.data import DataLoader, Dataset, random_split


# 1. Custom Dataset for Motion Classification
class MotionDataset(Dataset):
    """Dataset for motion classification (sudden vs normal)."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Array of shape (n_samples, seq_len, 3) containing (x, y, z) sequences
            labels: Array of shape (n_samples,) containing binary labels 
                    (0: normal/walking, 1: sudden motion)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# 2. Feature Engineering - Extract statistical features from raw data
class MotionFeatureExtractor:
    """
    Extract features from raw accelerometer sequences.
    Can be used to enhance the input data before feeding to the model.
    """
    
    @staticmethod
    def extract_features(data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from a sequence.
        
        Args:
            data: Shape (seq_len, 3) with columns [x, y, z]
            
        Returns:
            Feature array of shape (seq_len, n_features)
        """
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        
        # Compute additional features
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Jerk (rate of change)
        jerk_x = np.diff(x, prepend=x[0])
        jerk_y = np.diff(y, prepend=y[0])
        jerk_z = np.diff(z, prepend=z[0])
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        # Stack all features
        features = np.column_stack([
            x, y, z,              # Original 3
            magnitude,            # Total acceleration
            jerk_x, jerk_y, jerk_z,  # Jerk components
            jerk_magnitude,       # Total jerk
        ])
        
        return features.astype(np.float32)
    
    @staticmethod
    def extract_batch(sequences: np.ndarray) -> np.ndarray:
        """
        Extract features for a batch of sequences.
        
        Args:
            sequences: Shape (n_samples, seq_len, 3)
            
        Returns:
            Enhanced sequences of shape (n_samples, seq_len, n_features)
        """
        enhanced = []
        for seq in sequences:
            enhanced.append(MotionFeatureExtractor.extract_features(seq))
        return np.array(enhanced)


# 3. Lightning Data Module
class MotionDataModule(pl.LightningDataModule):
    """Data module for motion classification."""
    
    def __init__(
        self,
        sequences: np.ndarray = None,
        labels: np.ndarray = None,
        batch_size: int = 32,
        seq_len: int = 50,
        train_split: float = 0.7,
        val_split: float = 0.15,
        use_features: bool = True,  # Whether to extract additional features
    ):
        super().__init__()
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_split = train_split
        self.val_split = val_split
        self.use_features = use_features
        self.input_dim = 8 if use_features else 3  # 8 features or raw 3
        
    def prepare_data(self):
        """Prepare data - extract features if needed."""
        if self.sequences is not None and self.use_features:
            print("Extracting motion features...")
            self.sequences = MotionFeatureExtractor.extract_batch(self.sequences)
            self.input_dim = self.sequences.shape[2]
            print(f"Features extracted. Input dimension: {self.input_dim}")
    
    def setup(self, stage: Optional[str] = None):
        """Split data into train, val, test."""
        if self.sequences is None or self.labels is None:
            # Generate synthetic demo data
            print("No data provided. Generating synthetic demo data...")
            self.sequences, self.labels = self._generate_demo_data()
            if self.use_features:
                self.sequences = MotionFeatureExtractor.extract_batch(self.sequences)
                self.input_dim = self.sequences.shape[2]
        
        dataset = MotionDataset(self.sequences, self.labels)
        
        # Calculate split sizes
        n_total = len(dataset)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        n_test = n_total - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )
        
        print(f"Dataset splits - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    def _generate_demo_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic motion data for demonstration."""
        np.random.seed(42)
        
        sequences = []
        labels = []
        
        for i in range(n_samples):
            seq = np.zeros((self.seq_len, 3))
            
            if i % 2 == 0:  # Normal walking
                for t in range(self.seq_len):
                    phase = t * 0.1
                    seq[t, 0] = 0.1 * np.sin(phase) + np.random.normal(0, 0.05)
                    seq[t, 1] = 0.05 * np.cos(phase) + np.random.normal(0, 0.05)
                    seq[t, 2] = 1.0 + 0.1 * np.sin(phase) + np.random.normal(0, 0.05)
                labels.append(0)  # Normal
            else:  # Sudden motion
                spike_start = np.random.randint(10, 30)
                spike_len = np.random.randint(3, 8)
                
                for t in range(self.seq_len):
                    if spike_start <= t < spike_start + spike_len:
                        # Sudden spike
                        seq[t, 0] = np.random.uniform(1.5, 3.0) * np.random.choice([-1, 1])
                        seq[t, 1] = np.random.uniform(1.0, 2.5) * np.random.choice([-1, 1])
                        seq[t, 2] = np.random.uniform(0.5, 3.5)
                    else:
                        # Normal background
                        seq[t, 0] = np.random.normal(0, 0.1)
                        seq[t, 1] = np.random.normal(0, 0.1)
                        seq[t, 2] = 1.0 + np.random.normal(0, 0.1)
                labels.append(1)  # Sudden
        
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=4, persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4
        )


# 4. LSTM Model for Motion Classification
class MotionClassifierLSTM(pl.LightningModule):
    """
    LSTM-based classifier for sudden vs normal motion detection.
    
    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism (optional)
    - Classification head with dropout
    """
    
    def __init__(
        self,
        input_dim: int = 8,        # 8 features if using feature extraction
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        bidirectional: bool = True,
        use_attention: bool = True,
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
        
        # Attention layer (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Softmax(dim=1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )
        
        # Loss function with class weights (sudden events are often less frequent)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Attention mechanism
            attention_weights = self.attention(lstm_out)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            logits = self.classifier(context_vector)
        else:
            # Use last hidden state
            if self.hparams.bidirectional:
                last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                last_hidden = hidden[-1]
            logits = self.classifier(last_hidden)
        
        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
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
        
        self.training_step_outputs.append(
            {"loss": loss, "preds": preds, "labels": labels}
        )
        
        return loss
    
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        if outputs:
            all_preds = torch.cat([x["preds"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])
            epoch_acc = (all_preds == all_labels).float().mean()
            self.log("train_epoch_acc", epoch_acc)
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
            all_preds = torch.cat([x["preds"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])
            
            preds_np = all_preds.cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            
            f1 = f1_score(labels_np, preds_np, zero_division=0)
            self.log("val_f1", f1)
            
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.test_step_outputs.append({"loss": loss, "preds": preds, "labels": labels})
        return loss
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        if outputs:
            all_preds = torch.cat([x["preds"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])
            
            test_acc = (all_preds == all_labels).float().mean()
            
            preds_np = all_preds.cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            
            print("\n" + "=" * 50)
            print("MOTION CLASSIFICATION TEST RESULTS")
            print("=" * 50)
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision_score(labels_np, preds_np, zero_division=0):.4f}")
            print(f"Recall: {recall_score(labels_np, preds_np, zero_division=0):.4f}")
            print(f"F1-Score: {f1_score(labels_np, preds_np, zero_division=0):.4f}")
            print("\nClassification Report:")
            print(classification_report(
                labels_np, preds_np, 
                target_names=['Normal', 'Sudden'],
                zero_division=0
            ))
            print("\nConfusion Matrix:")
            print(confusion_matrix(labels_np, preds_np))
            print("=" * 50)
            
            self.test_step_outputs.clear()


# 5. Training Function
def train_motion_classifier(
    sequences: np.ndarray = None,
    labels: np.ndarray = None,
    max_epochs: int = 50,
    batch_size: int = 32,
    use_features: bool = True,
) -> Tuple[MotionClassifierLSTM, pl.Trainer, MotionDataModule]:
    """
    Train the motion classifier model.
    
    Args:
        sequences: Input sequences of shape (n_samples, seq_len, 3)
        labels: Binary labels (0: normal, 1: sudden)
        max_epochs: Maximum training epochs
        batch_size: Batch size
        use_features: Whether to extract additional features
        
    Returns:
        Trained model, trainer, and data module
    """
    # Initialize data module
    data_module = MotionDataModule(
        sequences=sequences,
        labels=labels,
        batch_size=batch_size,
        use_features=use_features,
    )
    
    # Prepare data to get input dimension
    data_module.prepare_data()
    
    # Initialize model
    model = MotionClassifierLSTM(
        input_dim=data_module.input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        bidirectional=True,
        use_attention=True,
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/motion/",
        filename="motion-classifier-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
    )
    
    # Logger
    logger = TensorBoardLogger("logs/", name="motion_classifier")
    
    # Trainer
    use_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator="gpu" if use_cuda else "cpu",
        devices=1 if use_cuda else None,
        log_every_n_steps=10,
        precision=16 if use_cuda else 32,
        logger=logger,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    return model, trainer, data_module


# 6. Inference Function
def predict_motion(
    model: MotionClassifierLSTM,
    sequence: np.ndarray,
    use_features: bool = True,
) -> Tuple[str, float]:
    """
    Predict if a sequence contains sudden motion.
    
    Args:
        model: Trained model
        sequence: Array of shape (seq_len, 3) containing (x, y, z)
        use_features: Whether to extract features (must match training)
        
    Returns:
        ('normal' or 'sudden'), confidence score
    """
    model.eval()
    
    # Extract features if needed
    if use_features:
        sequence = MotionFeatureExtractor.extract_features(sequence)
    
    with torch.no_grad():
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        device = next(model.parameters()).device
        sequence_tensor = sequence_tensor.to(device)
        
        logit = model(sequence_tensor)
        probability = torch.sigmoid(logit).item()
        
        if probability > 0.5:
            return 'sudden', probability
        else:
            return 'normal', 1 - probability


# 7. Create labels from rule-based detector (for training)
def generate_labels_from_rulebased(
    sequences: np.ndarray,
    threshold_confidence: float = 0.6,
) -> np.ndarray:
    """
    Use rule-based detector to generate training labels for ML model.
    
    This allows you to train the ML model using the rule-based detector
    as a "teacher" - useful when you don't have manually labeled data.
    
    Args:
        sequences: Array of shape (n_samples, seq_len, 3)
        threshold_confidence: Minimum confidence to label as sudden
        
    Returns:
        Labels array of shape (n_samples,)
    """
    # Import rule-based detector
    from detect_motion_rulebased import SuddenMotionDetector
    
    labels = []
    
    for seq in sequences:
        detector = SuddenMotionDetector(window_size=10)
        
        # Process entire sequence
        sudden_count = 0
        total_confidence = 0
        
        for t in range(len(seq)):
            result = detector.detect(seq[t, 0], seq[t, 1], seq[t, 2])
            if result.is_sudden:
                sudden_count += 1
                total_confidence += result.confidence
        
        # Label as sudden if enough sudden events with high confidence
        if sudden_count > 3 and (total_confidence / max(sudden_count, 1)) > threshold_confidence:
            labels.append(1)  # Sudden
        else:
            labels.append(0)  # Normal
    
    return np.array(labels, dtype=np.int64)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("MOTION CLASSIFIER - ML Approach (LSTM)")
    print("=" * 60)
    
    # Train model with synthetic data (replace with your actual data)
    print("\nTraining motion classifier...")
    model, trainer, data_module = train_motion_classifier(
        sequences=None,  # Will generate synthetic demo data
        labels=None,
        max_epochs=30,
        batch_size=32,
        use_features=True,
    )
    
    # Test prediction
    print("\nTesting prediction on sample sequence...")
    sample_seq_tensor, sample_label = data_module.test_dataset[0]
    sample_seq = sample_seq_tensor.cpu().numpy()
    
    # Note: For prediction, we need raw 3-channel data if use_features=True in training
    # Since demo data was already enhanced, create a raw test sample
    test_seq = np.random.randn(50, 3).astype(np.float32)  # Raw x,y,z
    prediction, confidence = predict_motion(model, test_seq, use_features=True)
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\n" + "=" * 60)
    print("To train on your own data:")
    print("  from detect_motion_ml import train_motion_classifier")
    print("  model, _, _ = train_motion_classifier(your_sequences, your_labels)")
    print("=" * 60)
