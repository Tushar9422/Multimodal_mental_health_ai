"""
Training script for audio emotion recognition model.

Trains CNN-LSTM model on RAVDESS dataset with:
- Train/validation split
- Class weighting for balanced training
- Early stopping and learning rate reduction
- Model checkpointing
- TensorBoard logging
"""

"""
Training script for audio emotion recognition model.
"""

# ========================================
# 🔧 GPU Configuration - MUST BE FIRST
# ========================================
import os

# Set before ANY TensorFlow import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.audio_emotion_model import create_audio_emotion_model
from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("train_audio")
config = get_config()


class AudioEmotionTrainer:
    """Trainer for audio emotion recognition model."""
    
    def __init__(self):
        self.config = config
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
        # Create output directories
        self.model_dir = ensure_dir(config.AUDIO_MODEL_DIR)
        self.checkpoint_dir = ensure_dir(config.CHECKPOINTS_DIR / "audio")
        self.tensorboard_dir = ensure_dir(config.TENSORBOARD_DIR / "audio")
        
        print("🎵 Audio Emotion Trainer Initialized")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")
    
    def load_data(self):
        """Load preprocessed audio features."""
        print("\n📂 Loading preprocessed audio data...")
        
        data_dir = config.DATA_DIR / "processed_features"
        
        # Load features and labels
        X = np.load(data_dir / "X_audio.npy")
        y = np.load(data_dir / "y_audio.npy")
        
        # Load emotion mapping
        with open(data_dir / "emotion_mapping.json", 'r') as f:
            self.emotion_mapping = json.load(f)
        
        print(f"✅ Data loaded successfully!")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Emotions: {self.emotion_mapping['emotions']}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepare data for training.
        
        Args:
            X: Features array
            y: Labels array
            test_size: Validation split ratio
        """
        print(f"\n🔄 Preparing data (test_size={test_size})...")
        
        # Convert labels to one-hot encoding
        num_classes = len(self.emotion_mapping['emotions'])
        y_categorical = keras.utils.to_categorical(y, num_classes)
        
        # Split into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y_categorical,
            test_size=test_size,
            random_state=42,
            stratify=y  # Keep class distribution
        )
        
        print(f"✅ Data split complete!")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Validation samples: {len(self.X_val)}")
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights(y)
        
        print(f"\n📊 Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for emotion_idx, count in zip(unique, counts):
            emotion = self.emotion_mapping['idx_to_emotion'][str(emotion_idx)]
            weight = self.class_weights.get(emotion_idx, 1.0)
            print(f"   {emotion}: {count} samples (weight: {weight:.2f})")
    
    def _calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data."""
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )
        
        class_weights = dict(zip(classes, weights))
        return class_weights
    
    def build_model(self):
        """Build and compile the model."""
        print("\n🏗️  Building model...")
        
        # Get input shape from training data
        input_shape = self.X_train.shape[1:]  # (features, time_steps)
        num_classes = self.y_train.shape[1]
        
        print(f"   Input shape: {input_shape}")
        print(f"   Number of classes: {num_classes}")
        
        # Create model
        self.model = create_audio_emotion_model(input_shape, num_classes)
        
        print("✅ Model built successfully!")
    
    def create_callbacks(self):
        """Create training callbacks."""
        print("\n⚙️  Setting up callbacks...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = []
        
        # 1. Model Checkpoint - Save best model
        checkpoint_path = self.checkpoint_dir / f"audio_model_{timestamp}.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"   ✅ Checkpoint: {checkpoint_path}")
        
        # 2. Early Stopping - Stop if no improvement
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"   ✅ Early stopping: patience=15")
        
        # 3. Reduce Learning Rate - Reduce LR when plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"   ✅ Reduce LR: factor=0.5, patience=5")
        
        # 4. TensorBoard - Visualization
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(self.tensorboard_dir / timestamp),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        print(f"   ✅ TensorBoard: {self.tensorboard_dir / timestamp}")
        
        # 5. CSV Logger - Save training history
        csv_logger = keras.callbacks.CSVLogger(
            str(self.model_dir / f"training_log_{timestamp}.csv")
        )
        callbacks.append(csv_logger)
        print(f"   ✅ CSV Logger enabled")
        
        return callbacks
    
    def train(self, epochs=50, batch_size=32):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print(f"\n🚀 Starting Training")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Validation samples: {len(self.X_val)}")
        print("="*60)
        
        callbacks = self.create_callbacks()
        
        with Timer("Model training"):
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=self.class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Training completed!")
    
    def evaluate(self):
        """Evaluate model on validation set."""
        print("\n📊 Evaluating model on validation set...")
        
        results = self.model.evaluate(
            self.X_val, self.y_val,
            batch_size=32,
            verbose=1
        )
        
        print("\n📈 Validation Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"   {metric_name}: {value:.4f}")
        
        return results
    
    def save_final_model(self):
        """Save the final trained model."""
        print("\n💾 Saving final model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"audio_emotion_final_{timestamp}.h5"
        
        self.model.save(model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save training history
        history_path = self.model_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            # Convert numpy values to Python types
            history_dict = {
                key: [float(val) for val in values]
                for key, values in self.history.history.items()
            }
            json.dump(history_dict, f, indent=2)
        
        print(f"✅ History saved: {history_path}")
        
        return model_path


def main():
    """Main training pipeline."""
    print("🎵 Audio Emotion Recognition - Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = AudioEmotionTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    # Prepare data
    trainer.prepare_data(X, y, test_size=0.2)
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train(
        epochs=50,
        batch_size=16
    )
    
    # Evaluate
    trainer.evaluate()
    
    # Save final model
    model_path = trainer.save_final_model()
    
    print("\n" + "="*60)
    print("🎉 Audio Model Training Complete!")
    print("="*60)
    print(f"📁 Model saved: {model_path}")
    print(f"📊 View training progress:")
    print(f"   tensorboard --logdir {trainer.tensorboard_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
