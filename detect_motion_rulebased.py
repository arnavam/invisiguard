"""
Rule-Based Motion Detection: Sudden vs Normal Fluctuation Detector

This module provides multiple rule-based methods to detect sudden motion fluctuations
versus normal walking/gradual movements in accelerometer data (x, y, z).

Methods included:
1. Vector Magnitude - Overall acceleration intensity
2. Jerk Magnitude - Rate of change of acceleration  
3. Rolling Window Statistics - Standard deviation over sliding window
4. Threshold-based detection - Combined approach with configurable thresholds
5. Optional Kalman Filter - For noise reduction if needed

Author: InvisiGuard Project
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class MotionState:
    """Represents the current motion state and detection results."""
    is_sudden: bool
    magnitude: float
    jerk: float
    rolling_std: float
    confidence: float  # 0-1, how confident we are in the detection
    motion_type: str  # 'sudden', 'normal', 'stationary'
    
    # Individual scores from all 4 methods (0-1 each)
    vector_magnitude_change: float = 0.0  # Method 1: √(Δx² + Δy² + Δz²)
    score_magnitude: float = 0.0          # Method 2: Threshold-based magnitude score
    score_jerk: float = 0.0               # Method 3: Jerk magnitude score
    score_rolling_std: float = 0.0        # Method 4: Rolling window std deviation score


class SimpleKalmanFilter:
    """
    1D Kalman Filter for noise reduction on sensor data.
    Use this if your accelerometer data is very noisy.
    """
    
    def __init__(self, process_variance: float = 1e-4, 
                 measurement_variance: float = 0.1,
                 initial_estimate: float = 0.0):
        """
        Args:
            process_variance: How much the true value changes (Q)
            measurement_variance: How noisy the sensor is (R)
            initial_estimate: Starting estimate value
        """
        self.q = process_variance  # Process variance
        self.r = measurement_variance  # Measurement variance
        self.x = initial_estimate  # Estimated value
        self.p = 1.0  # Estimation error covariance
        
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement and return filtered value.
        
        Args:
            measurement: New sensor reading
            
        Returns:
            Filtered (smoothed) value
        """
        # Prediction step
        self.p = self.p + self.q
        
        # Update step
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        
        return self.x
    
    def reset(self, initial_estimate: float = 0.0):
        """Reset filter state."""
        self.x = initial_estimate
        self.p = 1.0


class RollingWindow:
    """
    Rolling window for computing statistics over recent samples.
    This is NOT a smoother - it computes statistics to detect anomalies.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Number of samples to keep in the window
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def add(self, value: float) -> None:
        """Add a new value to the window."""
        self.buffer.append(value)
        
    def get_stats(self) -> Dict[str, float]:
        """
        Compute statistics over the current window.
        
        Returns:
            Dictionary with mean, std, min, max, range
        """
        if len(self.buffer) < 2:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0}
        
        arr = np.array(self.buffer)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'range': float(np.max(arr) - np.min(arr))
        }
    
    @property
    def std(self) -> float:
        """Get current standard deviation."""
        return self.get_stats()['std']
    
    @property
    def mean(self) -> float:
        """Get current mean."""
        return self.get_stats()['mean']
    
    def is_full(self) -> bool:
        """Check if window has enough samples."""
        return len(self.buffer) >= self.window_size


class SuddenMotionDetector:
    """
    Main class for detecting sudden vs normal motion fluctuations.
    
    Uses multiple methods combined:
    1. Vector Magnitude - sqrt(x² + y² + z²)
    2. Jerk - rate of change of acceleration
    3. Rolling Window Stats - std deviation to detect spikes
    4. Threshold Detection - configurable sensitivity
    
    Example Usage:
        detector = SuddenMotionDetector()
        
        # Real-time detection
        for x, y, z in accelerometer_stream:
            result = detector.detect(x, y, z)
            if result.is_sudden:
                print(f"Sudden motion detected! Type: {result.motion_type}")
    """
    
    # Default thresholds (tune these based on your sensor and use case)
    DEFAULT_THRESHOLDS = {
        'magnitude_sudden': 2.0,      # Above 2g is sudden (1g = normal gravity)
        'magnitude_normal': 1.2,      # Below 1.2g is normal walking
        'jerk_sudden': 5.0,           # Rapid change in acceleration
        'jerk_normal': 1.5,           # Normal jerk during walking
        'std_sudden': 1.5,            # High variance indicates sudden motion
        'std_normal': 0.5,            # Low variance during normal activity
        'stationary_threshold': 0.1,  # Below this is stationary
    }
    
    def __init__(self, 
                 window_size: int = 20,
                 use_kalman: bool = False,
                 kalman_process_var: float = 1e-4,
                 kalman_measurement_var: float = 0.1,
                 thresholds: Optional[Dict[str, float]] = None,
                 gravity: float = 9.81):
        """
        Initialize the sudden motion detector.
        
        Args:
            window_size: Size of rolling window for statistics
            use_kalman: Whether to apply Kalman filter for noise reduction
            kalman_process_var: Kalman filter process variance
            kalman_measurement_var: Kalman filter measurement variance
            thresholds: Custom thresholds (uses defaults if None)
            gravity: Gravity constant for normalization (9.81 m/s² or 1.0 if already in g)
        """
        self.window_size = window_size
        self.use_kalman = use_kalman
        self.gravity = gravity
        
        # Thresholds
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Rolling windows for each axis and magnitude
        self.window_x = RollingWindow(window_size)
        self.window_y = RollingWindow(window_size)
        self.window_z = RollingWindow(window_size)
        self.window_magnitude = RollingWindow(window_size)
        self.window_jerk = RollingWindow(window_size)
        
        # Kalman filters for each axis (if enabled)
        if use_kalman:
            self.kalman_x = SimpleKalmanFilter(kalman_process_var, kalman_measurement_var)
            self.kalman_y = SimpleKalmanFilter(kalman_process_var, kalman_measurement_var)
            self.kalman_z = SimpleKalmanFilter(kalman_process_var, kalman_measurement_var)
        
        # Previous values for jerk calculation
        self.prev_magnitude = None
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        
        # Detection history
        self.detection_history: List[MotionState] = []
        
    def calculate_magnitude(self, x: float, y: float, z: float) -> float:
        """
        Calculate vector magnitude (total acceleration).
        
        Formula: magnitude = √(x² + y² + z²)
        
        Args:
            x, y, z: Acceleration values in each axis
            
        Returns:
            Total acceleration magnitude
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    def calculate_jerk(self, current_magnitude: float) -> float:
        """
        Calculate jerk (rate of change of acceleration).
        
        Jerk = |a(t) - a(t-1)|
        
        Higher jerk = more sudden motion
        
        Args:
            current_magnitude: Current acceleration magnitude
            
        Returns:
            Jerk value (0 if first sample)
        """
        if self.prev_magnitude is None:
            return 0.0
        return abs(current_magnitude - self.prev_magnitude)
    
    def calculate_3d_jerk(self, x: float, y: float, z: float) -> float:
        """
        Calculate 3D jerk magnitude.
        
        Formula: jerk = √(Δx² + Δy² + Δz²)
        
        Args:
            x, y, z: Current acceleration values
            
        Returns:
            3D jerk magnitude
        """
        if self.prev_x is None:
            return 0.0
        
        dx = x - self.prev_x
        dy = y - self.prev_y
        dz = z - self.prev_z
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def detect(self, x: float, y: float, z: float) -> MotionState:
        """
        Main detection method - call this for each new sample.
        
        Args:
            x, y, z: Accelerometer readings (can be in m/s² or g units)
            
        Returns:
            MotionState with detection results
        """
        # Apply Kalman filter if enabled
        if self.use_kalman:
            x = self.kalman_x.update(x)
            y = self.kalman_y.update(y)
            z = self.kalman_z.update(z)
        
        # Normalize to g if input is in m/s²
        if self.gravity != 1.0:
            x_g = x / self.gravity
            y_g = y / self.gravity
            z_g = z / self.gravity
        else:
            x_g, y_g, z_g = x, y, z
        
        # Calculate magnitude
        magnitude = self.calculate_magnitude(x_g, y_g, z_g)
        
        # Calculate jerk (METHOD 2: Rate of change of acceleration)
        jerk = self.calculate_3d_jerk(x_g, y_g, z_g)
        
        # Calculate vector magnitude change (METHOD 1: √(Δx² + Δy² + Δz²))
        # This is the same as 3D jerk - measures change in acceleration vector
        vector_magnitude_change = jerk
        
        # Update rolling windows
        self.window_x.add(x_g)
        self.window_y.add(y_g)
        self.window_z.add(z_g)
        self.window_magnitude.add(magnitude)
        self.window_jerk.add(jerk)
        
        # Get rolling statistics (METHOD 4: Rolling Window Stats)
        rolling_std = self.window_magnitude.std
        
        # Determine motion type using ALL 4 methods
        is_sudden, motion_type, confidence, individual_scores = self._classify_motion(
            magnitude, jerk, rolling_std, vector_magnitude_change
        )
        
        # Update previous values
        self.prev_magnitude = magnitude
        self.prev_x = x_g
        self.prev_y = y_g
        self.prev_z = z_g
        
        # Create result with all 4 method scores
        result = MotionState(
            is_sudden=is_sudden,
            magnitude=magnitude,
            jerk=jerk,
            rolling_std=rolling_std,
            confidence=confidence,
            motion_type=motion_type,
            # Method scores
            vector_magnitude_change=vector_magnitude_change,
            score_magnitude=individual_scores['score_magnitude'],
            score_jerk=individual_scores['score_jerk'],
            score_rolling_std=individual_scores['score_rolling_std'],
        )
        
        # Store in history
        self.detection_history.append(result)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-500:]
        
        return result
    
    def _classify_motion(self, magnitude: float, jerk: float, 
                         rolling_std: float, vector_magnitude_change: float) -> Tuple[bool, str, float, Dict[str, float]]:
        """
        Classify motion using ALL 4 METHODS combined:
        
        1. Vector Magnitude Change: √(Δx² + Δy² + Δz²) - measures acceleration intensity change
        2. Jerk Magnitude: Rate of change of acceleration - sudden = high jerk
        3. Threshold-based: If magnitude exceeds threshold (>2g for sudden, <1g for walking)  
        4. Rolling Window Stats: Compare current window's std deviation vs baseline
        
        Returns:
            (is_sudden, motion_type, confidence, individual_scores)
        """
        t = self.thresholds
        
        # ============================================================
        # METHOD 1: Vector Magnitude Change Score (0-1)
        # √(Δx² + Δy² + Δz²) - Higher = more sudden
        # ============================================================
        if vector_magnitude_change > t['jerk_sudden']:  # Using jerk threshold
            score_vec_change = min(1.0, vector_magnitude_change / (t['jerk_sudden'] * 2))
        elif vector_magnitude_change < t['jerk_normal']:
            score_vec_change = max(0.0, vector_magnitude_change / t['jerk_sudden'] * 0.3)
        else:
            # Interpolate in between
            score_vec_change = 0.3 + 0.4 * (vector_magnitude_change - t['jerk_normal']) / (t['jerk_sudden'] - t['jerk_normal'])
        
        # ============================================================
        # METHOD 2: Jerk Magnitude Score (0-1)
        # Rate of change of acceleration - most important indicator
        # ============================================================
        if jerk > t['jerk_sudden']:
            score_jerk = min(1.0, 0.7 + 0.3 * (jerk - t['jerk_sudden']) / t['jerk_sudden'])
        elif jerk < t['jerk_normal']:
            score_jerk = max(0.0, jerk / t['jerk_normal'] * 0.3)
        else:
            # Interpolate in between
            score_jerk = 0.3 + 0.4 * (jerk - t['jerk_normal']) / (t['jerk_sudden'] - t['jerk_normal'])
        
        # ============================================================
        # METHOD 3: Threshold-based Magnitude Score (0-1)
        # Absolute acceleration level: >2g = sudden, <1.2g = normal
        # ============================================================
        if magnitude > t['magnitude_sudden']:
            score_magnitude = min(1.0, 0.7 + 0.3 * (magnitude - t['magnitude_sudden']) / t['magnitude_sudden'])
        elif magnitude < t['magnitude_normal']:
            score_magnitude = max(0.0, magnitude / t['magnitude_normal'] * 0.3)
        else:
            # Interpolate in between
            score_magnitude = 0.3 + 0.4 * (magnitude - t['magnitude_normal']) / (t['magnitude_sudden'] - t['magnitude_normal'])
        
        # ============================================================
        # METHOD 4: Rolling Window Statistics Score (0-1)
        # Standard deviation over window - high std = variable/sudden motion
        # ============================================================
        if rolling_std > t['std_sudden']:
            score_rolling = min(1.0, 0.7 + 0.3 * (rolling_std - t['std_sudden']) / t['std_sudden'])
        elif rolling_std < t['std_normal']:
            score_rolling = max(0.0, rolling_std / t['std_normal'] * 0.3)
        else:
            # Interpolate in between
            score_rolling = 0.3 + 0.4 * (rolling_std - t['std_normal']) / (t['std_sudden'] - t['std_normal'])
        
        # Store individual scores
        individual_scores = {
            'vector_magnitude_change': score_vec_change,
            'score_magnitude': score_magnitude,
            'score_jerk': score_jerk,
            'score_rolling_std': score_rolling,
        }
        
        # ============================================================
        # COMBINED SCORING with weights
        # Jerk is most important, then magnitude, then rolling std
        # ============================================================
        weights = {
            'vector_magnitude_change': 0.25,  # 25% weight
            'magnitude': 0.25,                # 25% weight  
            'jerk': 0.30,                     # 30% weight (most important)
            'rolling_std': 0.20,              # 20% weight
        }
        
        combined_score = (
            weights['vector_magnitude_change'] * score_vec_change +
            weights['magnitude'] * score_magnitude +
            weights['jerk'] * score_jerk +
            weights['rolling_std'] * score_rolling
        )
        
        # Stationary check
        if magnitude < t['stationary_threshold']:
            return False, 'stationary', 0.9, individual_scores
        
        # Classification: Above 0.5 = sudden, below = normal
        if combined_score > 0.5:
            return True, 'sudden', combined_score, individual_scores
        else:
            return False, 'normal', 1 - combined_score, individual_scores
    
    def detect_batch(self, data: np.ndarray) -> List[MotionState]:
        """
        Detect sudden motion in a batch of data.
        
        Args:
            data: Array of shape (N, 3) with columns [x, y, z]
            
        Returns:
            List of MotionState for each sample
        """
        results = []
        for i in range(len(data)):
            x, y, z = data[i]
            results.append(self.detect(x, y, z))
        return results
    
    def reset(self):
        """Reset all internal state."""
        self.window_x = RollingWindow(self.window_size)
        self.window_y = RollingWindow(self.window_size)
        self.window_z = RollingWindow(self.window_size)
        self.window_magnitude = RollingWindow(self.window_size)
        self.window_jerk = RollingWindow(self.window_size)
        
        if self.use_kalman:
            self.kalman_x.reset()
            self.kalman_y.reset()
            self.kalman_z.reset()
        
        self.prev_magnitude = None
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.detection_history = []
    
    def get_summary(self) -> Dict:
        """Get summary of recent detections."""
        if not self.detection_history:
            return {'total': 0, 'sudden': 0, 'normal': 0, 'stationary': 0}
        
        recent = self.detection_history[-100:]  # Last 100 samples
        
        return {
            'total': len(recent),
            'sudden': sum(1 for r in recent if r.motion_type == 'sudden'),
            'normal': sum(1 for r in recent if r.motion_type == 'normal'),
            'stationary': sum(1 for r in recent if r.motion_type == 'stationary'),
            'avg_magnitude': np.mean([r.magnitude for r in recent]),
            'avg_jerk': np.mean([r.jerk for r in recent]),
            'max_jerk': max(r.jerk for r in recent),
        }


def analyze_file(file_path: str, 
                 x_col: str = 'x', 
                 y_col: str = 'y', 
                 z_col: str = 'z',
                 use_kalman: bool = False) -> Tuple[List[MotionState], Dict]:
    """
    Analyze a CSV file for sudden motion events.
    
    Args:
        file_path: Path to CSV file
        x_col, y_col, z_col: Column names for accelerometer axes
        use_kalman: Whether to use Kalman filter
        
    Returns:
        List of detection results and summary statistics
    """
    import pandas as pd
    
    df = pd.read_csv(file_path)
    
    # Find accelerometer columns (flexible naming)
    possible_x = ['x', 'acc_x', 'accel_x', 'ax', 'X']
    possible_y = ['y', 'acc_y', 'accel_y', 'ay', 'Y']
    possible_z = ['z', 'acc_z', 'accel_z', 'az', 'Z']
    
    x_col = next((c for c in possible_x if c in df.columns), x_col)
    y_col = next((c for c in possible_y if c in df.columns), y_col)
    z_col = next((c for c in possible_z if c in df.columns), z_col)
    
    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        raise ValueError(f"Columns not found. Available: {list(df.columns)}")
    
    # Create detector
    detector = SuddenMotionDetector(use_kalman=use_kalman)
    
    # Process data
    data = df[[x_col, y_col, z_col]].values
    results = detector.detect_batch(data)
    
    # Get summary
    summary = detector.get_summary()
    
    # Find sudden motion events
    sudden_events = []
    for i, r in enumerate(results):
        if r.is_sudden:
            sudden_events.append({
                'index': i,
                'magnitude': r.magnitude,
                'jerk': r.jerk,
                'confidence': r.confidence
            })
    
    summary['sudden_events'] = sudden_events
    summary['sudden_event_count'] = len(sudden_events)
    
    return results, summary


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("SUDDEN MOTION DETECTOR - Rule-Based Approach")
    print("=" * 60)
    
    # Create detector with default settings
    detector = SuddenMotionDetector(
        window_size=20,
        use_kalman=False,  # Set to True for noisy data
        gravity=1.0  # Set to 9.81 if data is in m/s²
    )
    
    # Simulate different motion patterns
    print("\n1. Simulating NORMAL WALKING pattern...")
    np.random.seed(42)
    
    # Normal walking: small rhythmic variations around 1g
    walking_data = []
    for i in range(100):
        t = i * 0.02  # 50Hz sampling
        x = 0.1 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.05)
        y = 0.05 * np.cos(2 * np.pi * 2 * t) + np.random.normal(0, 0.05)
        z = 1.0 + 0.1 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.05)
        walking_data.append((x, y, z))
        result = detector.detect(x, y, z)
    
    walking_summary = detector.get_summary()
    print(f"   Sudden events: {walking_summary['sudden']}")
    print(f"   Normal events: {walking_summary['normal']}")
    print(f"   Avg magnitude: {walking_summary['avg_magnitude']:.3f}g")
    
    # Reset for next test
    detector.reset()
    
    print("\n2. Simulating SUDDEN MOTION (fall/impact)...")
    
    # Sudden motion: large spike followed by oscillations
    sudden_data = []
    for i in range(100):
        t = i * 0.02
        if 30 <= i <= 35:  # Sudden spike
            x = 2.5 + np.random.normal(0, 0.3)
            y = -1.5 + np.random.normal(0, 0.3)
            z = 3.0 + np.random.normal(0, 0.3)
        elif 36 <= i <= 50:  # Post-impact oscillation
            factor = (50 - i) / 14
            x = factor * np.sin(10 * t) + np.random.normal(0, 0.1)
            y = factor * np.cos(10 * t) + np.random.normal(0, 0.1)
            z = 1.0 + factor * np.sin(10 * t) + np.random.normal(0, 0.1)
        else:  # Normal
            x = np.random.normal(0, 0.1)
            y = np.random.normal(0, 0.1)
            z = 1.0 + np.random.normal(0, 0.1)
        
        sudden_data.append((x, y, z))
        result = detector.detect(x, y, z)
        
        if result.is_sudden:
            print(f"   [Sample {i}] SUDDEN detected!")
            print(f"      Scores: VecChange={result.vector_magnitude_change:.2f}, "
                  f"Mag={result.score_magnitude:.2f}, Jerk={result.score_jerk:.2f}, "
                  f"RollingStd={result.score_rolling_std:.2f}")
            print(f"      Combined Confidence: {result.confidence:.2%}")
    
    sudden_summary = detector.get_summary()
    print(f"\n   Total sudden events: {sudden_summary['sudden']}")
    print(f"   Max jerk observed: {sudden_summary['max_jerk']:.3f}")
    
    print("\n" + "=" * 60)
    print("Detection complete. Use SuddenMotionDetector class in your app.")
    print("=" * 60)
    
    # Example with file
    print("\n\nTo analyze a CSV file, use:")
    print("  results, summary = analyze_file('your_data.csv')")
    print("  print(f'Found {summary[\"sudden_event_count\"]} sudden events')")
