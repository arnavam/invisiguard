"""
Activity Shift Detector

This module detects transitions between different activities (e.g., running → jumping)
using the LSTM classifier. It maintains temporal state and identifies when the
dominant activity changes.

Key Features:
- Temporal smoothing to avoid false alarms from momentary misclassifications
- Configurable persistence window for shift confirmation
- Returns shift events with timestamps and activity names
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from lstm_encoder_decoder import LSTMEventClassifier


@dataclass
class ShiftEvent:
    """Represents a detected activity shift."""
    timestamp: datetime
    from_activity: str
    to_activity: str
    confidence: float
    window_index: int
    
    def __str__(self) -> str:
        return (f"[{self.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"SHIFT: {self.from_activity} → {self.to_activity} "
                f"(confidence: {self.confidence:.2%})")


@dataclass 
class BinaryShiftEvent:
    """Simple binary shift event - just indicates a shift occurred."""
    timestamp: datetime
    confidence: float
    window_index: int
    from_class: int  # numeric class ID
    to_class: int    # numeric class ID
    
    def __str__(self) -> str:
        return (f"[{self.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"SHIFT DETECTED (class {self.from_class} → {self.to_class}, "
                f"confidence: {self.confidence:.2%})")


class ActivityShiftDetector:
    """
    Detects shifts between different activities using temporal tracking.
    
    The detector maintains a sliding window of recent classifications and
    identifies when the dominant activity changes persistently.
    """
    
    def __init__(
        self,
        classifier: LSTMEventClassifier,
        label_map: Optional[Dict[str, int]] = None,
        persistence_window: int = 5,
        min_confidence: float = 0.6,
        smoothing_window: int = 3,
    ):
        """
        Args:
            classifier: Trained LSTMEventClassifier model
            label_map: Optional dictionary mapping activity names to label indices.
                       If not provided, auto-generates labels like "activity_0", "activity_1", etc.
            persistence_window: Number of consecutive windows needed to confirm shift
            min_confidence: Minimum confidence to consider a prediction valid
            smoothing_window: Size of sliding window for activity smoothing
        """
        self.classifier = classifier
        self.classifier.eval()
        
        # Auto-generate label map if not provided
        if label_map is None:
            num_classes = classifier.num_classes
            label_map = {f"activity_{i}": i for i in range(num_classes)}
            print(f"Auto-generated labels: {list(label_map.keys())}")
        
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        
        self.persistence_window = persistence_window
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        
        # State tracking
        self.current_activity: Optional[str] = None
        self.activity_history: deque = deque(maxlen=smoothing_window)
        self.pending_shift: Optional[Tuple[str, int]] = None  # (activity, count)
        self.shift_events: List[ShiftEvent] = []
        self.window_count = 0
        
    def reset(self):
        """Reset detector state."""
        self.current_activity = None
        self.activity_history.clear()
        self.pending_shift = None
        self.shift_events = []
        self.window_count = 0
        
    def _get_dominant_activity(self) -> Optional[str]:
        """Get the most common activity in the history window."""
        if not self.activity_history:
            return None
        
        # Count occurrences of each activity
        activity_counts: Dict[str, int] = {}
        for activity, _ in self.activity_history:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Return the most common
        dominant = max(activity_counts.items(), key=lambda x: x[1])
        return dominant[0]
    
    def _classify_sequence(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single sequence.
        
        Args:
            sequence: Input of shape (seq_len, n_features)
            
        Returns:
            activity_name: Predicted activity name
            confidence: Prediction confidence
        """
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dim
            preds, confs = self.classifier.predict(x)
            pred_class = int(preds.item())
            confidence = float(confs.item())
            
        activity_name = self.reverse_label_map.get(pred_class, f"unknown_{pred_class}")
        return activity_name, confidence
    
    def process_window(
        self,
        sequence: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ShiftEvent]:
        """
        Process a single time window and check for activity shift.
        
        Args:
            sequence: Input sequence of shape (seq_len, n_features)
            timestamp: Optional timestamp for this window
            
        Returns:
            ShiftEvent if a shift was detected, None otherwise
        """
        timestamp = timestamp or datetime.now()
        self.window_count += 1
        
        # Classify the current window
        activity, confidence = self._classify_sequence(sequence)
        
        # Skip low-confidence predictions
        if confidence < self.min_confidence:
            return None
        
        # Add to history
        self.activity_history.append((activity, confidence))
        
        # Get smoothed (dominant) activity
        dominant_activity = self._get_dominant_activity()
        
        # Initialize current activity if first window
        if self.current_activity is None:
            self.current_activity = dominant_activity
            return None
        
        # Check if dominant activity differs from current
        if dominant_activity != self.current_activity:
            # Start or continue pending shift
            if self.pending_shift is None or self.pending_shift[0] != dominant_activity:
                self.pending_shift = (dominant_activity, 1)
            else:
                self.pending_shift = (dominant_activity, self.pending_shift[1] + 1)
            
            # Check if shift is confirmed
            if self.pending_shift[1] >= self.persistence_window:
                # Calculate average confidence for the shift
                recent_confs = [
                    conf for act, conf in self.activity_history 
                    if act == dominant_activity
                ]
                avg_confidence = np.mean(recent_confs) if recent_confs else confidence
                
                # Create shift event
                shift_event = ShiftEvent(
                    timestamp=timestamp,
                    from_activity=self.current_activity,
                    to_activity=dominant_activity,
                    confidence=avg_confidence,
                    window_index=self.window_count,
                )
                
                # Update state
                self.current_activity = dominant_activity
                self.pending_shift = None
                self.shift_events.append(shift_event)
                
                return shift_event
        else:
            # Activity matches current - cancel any pending shift
            self.pending_shift = None
            
        return None
    
    def process_stream(
        self,
        sequences: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ) -> List[ShiftEvent]:
        """
        Process a stream of sequences and detect all shifts.
        
        Args:
            sequences: Array of shape (n_windows, seq_len, n_features)
            timestamps: Optional list of timestamps for each window
            
        Returns:
            List of detected ShiftEvent objects
        """
        self.reset()
        detected_shifts = []
        
        for i, seq in enumerate(sequences):
            ts = timestamps[i] if timestamps else None
            shift = self.process_window(seq, ts)
            if shift:
                detected_shifts.append(shift)
                print(shift)
        
        return detected_shifts
    
    def get_current_state(self) -> Dict:
        """Get current detector state for debugging/monitoring."""
        return {
            "current_activity": self.current_activity,
            "window_count": self.window_count,
            "history_size": len(self.activity_history),
            "pending_shift": self.pending_shift,
            "total_shifts_detected": len(self.shift_events),
        }
    
    def get_activity_timeline(self) -> List[Tuple[int, str, float]]:
        """
        Get a timeline of activity classifications.
        
        Returns:
            List of (window_index, activity, confidence) tuples
        """
        # Note: This only contains recent history due to deque maxlen
        return [
            (self.window_count - len(self.activity_history) + i, act, conf)
            for i, (act, conf) in enumerate(self.activity_history)
        ]


# ============================================================================
# Streaming Processor for Real-time Use
# ============================================================================
class RealTimeShiftDetector:
    """
    Real-time wrapper that processes continuous sensor data.
    
    This class handles buffering and windowing of raw sensor data,
    then feeds windows to the ActivityShiftDetector.
    """
    
    def __init__(
        self,
        shift_detector: ActivityShiftDetector,
        seq_len: int = 100,
        step_size: int = 20,
    ):
        """
        Args:
            shift_detector: ActivityShiftDetector instance
            seq_len: Length of each sequence window
            step_size: Step size between windows (overlap = seq_len - step_size)
        """
        self.shift_detector = shift_detector
        self.seq_len = seq_len
        self.step_size = step_size
        
        # Buffer for incoming data
        self.buffer: List[np.ndarray] = []
        self.buffer_position = 0
        
    def reset(self):
        """Reset the processor state."""
        self.buffer = []
        self.buffer_position = 0
        self.shift_detector.reset()
        
    def add_sample(self, sample: np.ndarray) -> Optional[ShiftEvent]:
        """
        Add a single sample and process if window is ready.
        
        Args:
            sample: Single timestep of shape (n_features,)
            
        Returns:
            ShiftEvent if a shift was detected, None otherwise
        """
        self.buffer.append(sample)
        
        # Check if we have enough data for a window
        if len(self.buffer) >= self.seq_len:
            # Extract window
            window = np.array(self.buffer[:self.seq_len])
            
            # Process window
            shift = self.shift_detector.process_window(window)
            
            # Slide the buffer
            self.buffer = self.buffer[self.step_size:]
            
            return shift
        
        return None
    
    def add_samples(self, samples: np.ndarray) -> List[ShiftEvent]:
        """
        Add multiple samples at once.
        
        Args:
            samples: Array of shape (n_samples, n_features)
            
        Returns:
            List of any detected ShiftEvents
        """
        shifts = []
        for sample in samples:
            shift = self.add_sample(sample)
            if shift:
                shifts.append(shift)
        return shifts


# ============================================================================
# Binary Shift Detector - Simple Yes/No Shift Detection
# ============================================================================
class BinaryShiftDetector:
    """
    Simple binary shift detector - just detects if ANY activity shift occurred.
    
    No need to know activity names - just outputs True when the classifier's
    prediction changes from one class to another.
    """
    
    def __init__(
        self,
        classifier: LSTMEventClassifier,
        persistence_window: int = 5,
        min_confidence: float = 0.6,
        smoothing_window: int = 3,
    ):
        """
        Args:
            classifier: Trained LSTMEventClassifier model
            persistence_window: Number of consecutive windows needed to confirm shift
            min_confidence: Minimum confidence to consider a prediction valid
            smoothing_window: Size of sliding window for class smoothing
        """
        self.classifier = classifier
        self.classifier.eval()
        
        self.persistence_window = persistence_window
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window
        
        # State tracking (uses numeric class IDs, not names)
        self.current_class: Optional[int] = None
        self.class_history: deque = deque(maxlen=smoothing_window)
        self.pending_shift: Optional[Tuple[int, int]] = None  # (class_id, count)
        self.shift_events: List[BinaryShiftEvent] = []
        self.window_count = 0
        
    def reset(self):
        """Reset detector state."""
        self.current_class = None
        self.class_history.clear()
        self.pending_shift = None
        self.shift_events = []
        self.window_count = 0
        
    def _get_dominant_class(self) -> Optional[int]:
        """Get the most common class in the history window."""
        if not self.class_history:
            return None
        
        # Count occurrences of each class
        class_counts: Dict[int, int] = {}
        for class_id, _ in self.class_history:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Return the most common
        dominant = max(class_counts.items(), key=lambda x: x[1])
        return dominant[0]
    
    def _classify_sequence(self, sequence: np.ndarray) -> Tuple[int, float]:
        """Classify a single sequence, returns (class_id, confidence)."""
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0)
            preds, confs = self.classifier.predict(x)
            return int(preds.item()), float(confs.item())
    
    def process_window(
        self,
        sequence: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[BinaryShiftEvent]]:
        """
        Process a single time window.
        
        Args:
            sequence: Input sequence of shape (seq_len, n_features)
            timestamp: Optional timestamp for this window
            
        Returns:
            (shift_detected, shift_event): Boolean and optional event details
        """
        timestamp = timestamp or datetime.now()
        self.window_count += 1
        
        # Classify
        class_id, confidence = self._classify_sequence(sequence)
        
        # Skip low-confidence
        if confidence < self.min_confidence:
            return False, None
        
        # Add to history
        self.class_history.append((class_id, confidence))
        
        # Get dominant class
        dominant_class = self._get_dominant_class()
        
        # Initialize if first window
        if self.current_class is None:
            self.current_class = dominant_class
            return False, None
        
        # Check for shift
        if dominant_class != self.current_class:
            # Start or continue pending shift
            if self.pending_shift is None or self.pending_shift[0] != dominant_class:
                self.pending_shift = (dominant_class, 1)
            else:
                self.pending_shift = (dominant_class, self.pending_shift[1] + 1)
            
            # Confirm shift?
            if self.pending_shift[1] >= self.persistence_window:
                recent_confs = [
                    conf for cid, conf in self.class_history 
                    if cid == dominant_class
                ]
                avg_confidence = np.mean(recent_confs) if recent_confs else confidence
                
                shift_event = BinaryShiftEvent(
                    timestamp=timestamp,
                    confidence=avg_confidence,
                    window_index=self.window_count,
                    from_class=self.current_class,
                    to_class=dominant_class,
                )
                
                self.current_class = dominant_class
                self.pending_shift = None
                self.shift_events.append(shift_event)
                
                return True, shift_event
        else:
            self.pending_shift = None
            
        return False, None
    
    def process_stream(
        self,
        sequences: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ) -> List[BinaryShiftEvent]:
        """
        Process a stream of sequences.
        
        Args:
            sequences: Array of shape (n_windows, seq_len, n_features)
            timestamps: Optional list of timestamps
            
        Returns:
            List of detected BinaryShiftEvent objects
        """
        self.reset()
        detected_shifts = []
        
        for i, seq in enumerate(sequences):
            ts = timestamps[i] if timestamps else None
            shift_detected, event = self.process_window(seq, ts)
            if shift_detected:
                detected_shifts.append(event)
                print(f"SHIFT DETECTED at window {self.window_count}")
        
        return detected_shifts
    
    def get_shift_count(self) -> int:
        """Get total number of shifts detected."""
        return len(self.shift_events)
    
    def had_any_shift(self) -> bool:
        """Simple check: did any shift occur?"""
        return len(self.shift_events) > 0


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("Activity Shift Detector - Demo")
    print("=" * 60)
    
    # Check if a trained model exists
    checkpoint_dir = "checkpoints/classifier"
    
    if not os.path.exists(checkpoint_dir):
        print("\nNo trained classifier found!")
        print("Please train a classifier first using lstm_encoder_decoder.py")
        print("\nExample usage after training:")
        print("""
    from lstm_encoder_decoder import LSTMEventClassifier
    from shift_detector import ActivityShiftDetector
    
    # Load your trained model
    classifier = LSTMEventClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")
    
    # Define your activity labels
    label_map = {"walking": 0, "running": 1, "jumping": 2, "standing": 3}
    
    # Create shift detector
    detector = ActivityShiftDetector(
        classifier=classifier,
        label_map=label_map,
        persistence_window=5,  # Require 5 consecutive windows to confirm shift
        min_confidence=0.6,    # Ignore low-confidence predictions
    )
    
    # Process a stream of data
    # sequences shape: (n_windows, seq_len, n_features)
    shifts = detector.process_stream(sequences)
    
    for shift in shifts:
        print(shift)
    """)
    else:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not checkpoints:
            print("No checkpoint files found in", checkpoint_dir)
        else:
            latest_ckpt = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"\nFound checkpoint: {latest_ckpt}")
            
            # Load model and create detector
            print("\nLoading classifier...")
            classifier = LSTMEventClassifier.load_from_checkpoint(latest_ckpt)
            
            # For demo, using fall detection labels
            label_map = {"adl": 0, "fall": 1}
            
            detector = ActivityShiftDetector(
                classifier=classifier,
                label_map=label_map,
                persistence_window=3,
                min_confidence=0.5,
            )
            
            # Generate synthetic test data for demo
            print("\nGenerating synthetic test data...")
            np.random.seed(42)
            
            # Simulate: 20 windows of "normal", then 20 windows of "event"
            n_windows = 40
            seq_len = 100
            n_features = 3
            
            # Create synthetic data with clear pattern change
            sequences = np.zeros((n_windows, seq_len, n_features))
            
            for i in range(n_windows):
                if i < 20:
                    # "Normal" activity - smooth sine waves
                    t = np.linspace(0, 4*np.pi, seq_len)
                    sequences[i, :, 0] = np.sin(t) + np.random.randn(seq_len) * 0.1
                    sequences[i, :, 1] = np.cos(t) + np.random.randn(seq_len) * 0.1
                    sequences[i, :, 2] = np.random.randn(seq_len) * 0.2
                else:
                    # "Event" activity - high amplitude, erratic
                    sequences[i] = np.random.randn(seq_len, n_features) * 2
            
            print(f"Test data shape: {sequences.shape}")
            
            # Process stream
            print("\nProcessing stream for shifts...\n")
            shifts = detector.process_stream(sequences)
            
            print(f"\nTotal shifts detected: {len(shifts)}")
            print(f"Final state: {detector.get_current_state()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
