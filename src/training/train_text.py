#!/usr/bin/env python3
"""
Training script for text sentiment analysis model.
Fine-tunes DistilBERT on mental health text data.

FIXED FOR: TensorFlow 2.16.2+ GPU compatibility
"""

# ========================================
# 🔧 GPU FIX: Must be BEFORE TensorFlow import
# ========================================
import os

# GPU environment fixes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🔧 GPU environment configured for text model training")

# ========================================
# NOW import TensorFlow
# ========================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.text_sentiment_model import TextSentimentModel
from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("train_text")
config = get_config()


class TextSentimentTrainer:
    """Trainer for text sentiment analysis model."""
    
    def __init__(self):
        self.config = config
        self.model_builder = None
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.metadata = None
        
        # Create output directories
        self.model_dir = ensure_dir(config.TEXT_MODEL_DIR)
        self.checkpoint_dir = ensure_dir(config.CHECKPOINTS_DIR / "text")
        self.tensorboard_dir = ensure_dir(config.TENSORBOARD_DIR / "text")
        
        print("💬 Text Sentiment Trainer Initialized")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")
    
    def load_data(self):
        """Load preprocessed text data."""
        print("\n📂 Loading preprocessed text data...")
        
        data_dir = config.DATA_DIR / "processed_features"
        
        # Load data splits
        self.X_train = np.load(data_dir / "X_text_train.npy", allow_pickle=True)
        self.X_val = np.load(data_dir / "X_text_val.npy", allow_pickle=True)
        self.y_train = np.load(data_dir / "y_text_train.npy")
        self.y_val = np.load(data_dir / "y_text_val.npy")
        
        # Load metadata
        with open(data_dir / "text_metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Validation samples: {len(self.X_val)}")
        print(f"   Categories: {self.metadata['categories']}")
        
        return True
    
    def calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        print("\n⚖️  Calculating class weights...")
        
        # Get unique classes and compute weights
        classes = np.unique(self.y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.y_train
        )
        
        class_weights = dict(zip(classes.astype(int), weights))
        
        print(f"   Class weights:")
        for class_idx, weight in class_weights.items():
            category = self.metadata['idx_to_category'][str(class_idx)]
            count = np.sum(self.y_train == class_idx)
            print(f"   {category:15s}: {count:5d} samples (weight: {weight:.3f})")
        
        return class_weights
    
    def build_model(self):
        """Build and compile the model."""
        print("\n🏗️  Building model...")
        
        num_classes = self.metadata['num_classes']
        max_length = self.metadata['max_length']
        
        print(f"   Number of classes: {num_classes}")
        print(f"   Max sequence length: {max_length}")
        
        # Create model
        self.model_builder = TextSentimentModel(
            num_classes=num_classes,
            max_tokens=10000,
            max_length=max_length
        )
        
        # Create vectorizer with training data
        self.model_builder.create_vectorizer(self.X_train)
        
        # Build and compile
        self.model = self.model_builder.build_model()
        self.model_builder.compile_model(learning_rate=0.001)
        
        print("✅ Model built successfully!")

    
    def create_tf_dataset(self, X, y, batch_size=32, shuffle=True):
        """
        Create TensorFlow dataset.
        BiLSTM model takes raw text directly.
        """
        print(f"   Creating dataset with {len(X)} samples...")
        
        # FIX: Don't reshape - keep as 1D array of strings
        # Model expects input shape (batch_size,) not (batch_size, 1)
        
        # Create dataset from string array
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset



    
    def create_callbacks(self):
        """Create training callbacks."""
        print("\n⚙️  Setting up callbacks...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = []
        
        # 1. Model Checkpoint
        checkpoint_path = self.checkpoint_dir / f"text_model_{timestamp}.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"   ✅ Checkpoint: {checkpoint_path}")
        
        # 2. Early Stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,  # DistilBERT converges fast
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"   ✅ Early stopping: patience=3")
        
        # 3. Reduce Learning Rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"   ✅ Reduce LR: factor=0.5, patience=2")
        
        # 4. TensorBoard
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(self.tensorboard_dir / timestamp),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        print(f"   ✅ TensorBoard: enabled")
        
        # 5. CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            str(self.model_dir / f"training_log_{timestamp}.csv")
        )
        callbacks.append(csv_logger)
        print(f"   ✅ CSV Logger: enabled")
        
        return callbacks
    
    def train(self, epochs=5, batch_size=16):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs (DistilBERT needs few epochs)
            batch_size: Batch size for training
        """
        print(f"\n🚀 Starting Training")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Validation samples: {len(self.X_val)}")
        print("="*60)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights()
        
        # Create TensorFlow datasets
        print("\n📦 Creating TensorFlow datasets...")
        train_dataset = self.create_tf_dataset(
            self.X_train, self.y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataset = self.create_tf_dataset(
            self.X_val, self.y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        print("✅ Datasets created!")
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train
        print("\n🎯 Training DistilBERT...")
        with Timer("Model training"):
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Training completed!")
    
    def evaluate(self):
        """Evaluate model on validation set."""
        print("\n📊 Evaluating model on validation set...")
        
        # Create validation dataset
        val_dataset = self.create_tf_dataset(
            self.X_val, self.y_val,
            batch_size=32,
            shuffle=False
        )
        
        results = self.model.evaluate(val_dataset, verbose=1)
        
        print("\n📈 Validation Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"   {metric_name}: {value:.4f}")
        
        return results
    
    def save_final_model(self):
        """Save the final trained model."""
        print("\n💾 Saving final model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"text_sentiment_final_{timestamp}.h5"
        
        self.model.save(model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save training history
        history_path = self.model_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            history_dict = {
                key: [float(val) for val in values]
                for key, values in self.history.history.items()
            }
            json.dump(history_dict, f, indent=2)
        
        print(f"✅ History saved: {history_path}")
        
        return model_path


def main():
    """Main training pipeline."""
    print("💬 Text Sentiment Analysis - Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = TextSentimentTrainer()
    
    # Load data
    trainer.load_data()
    
    # Build model
    trainer.build_model()
    
    # Train model (DistilBERT converges fast, only need 5 epochs)
    trainer.train(
        epochs=5,
        batch_size=16  # Conservative batch size for stability
    )
    
    # Evaluate
    trainer.evaluate()
    
    # Save final model
    model_path = trainer.save_final_model()
    
    print("\n" + "="*60)
    print("🎉 Text Model Training Complete!")
    print("="*60)
    print(f"📁 Model saved: {model_path}")
    print(f"📊 View training progress:")
    print(f"   tensorboard --logdir {trainer.tensorboard_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
