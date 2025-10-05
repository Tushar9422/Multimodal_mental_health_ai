"""
Helper utilities optimized for WSL2 Ubuntu environment.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Tuple
from datetime import datetime

def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """Save dictionary as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file as dictionary."""
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_duration(seconds: float) -> str:
    """Format duration to human readable."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {seconds%60:.1f}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
        
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        print(f"{self.name} completed in {format_duration(duration)}")

def check_wsl2_gpu() -> Dict[str, Any]:
    """Check GPU availability in WSL2."""
    gpu_info = {'available': False, 'devices': []}
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info['available'] = len(gpus) > 0
        gpu_info['devices'] = [device.name for device in gpus]
        gpu_info['count'] = len(gpus)
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info

def normalize_audio_features(features: np.ndarray) -> np.ndarray:
    """Normalize audio features."""
    return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

def prepare_training_data(X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Tuple[np.ndarray, ...]:
    """Split data for training."""
    n_samples = len(X)
    n_train = int(n_samples * (1 - validation_split))
    indices = np.random.permutation(n_samples)
    train_indices, val_indices = indices[:n_train], indices[n_train:]
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
