"""
Training script for facial expression recognition model.
Uses EfficientNet-B0 transfer learning on FER-2013 dataset.

FIXED FOR: TensorFlow 2.16.2+ GPU compatibility
"""

# ========================================
# 🔧 GPU FIX: Must be BEFORE TensorFlow import
# ========================================
import os

# GPU environment fixes (from audio model lessons learned)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🔧 GPU environment configured for facial model training")

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

from src.models.facial_emotion_model import FacialEmotionModel
from src.preprocessing.image_processor import FER2013DataLoader
from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("train_facial")
config = get_config()


class FacialEmotionTrainer:
    """Trainer for facial expression recognition model."""
    
    def __init__(self):
        self.config = config
        self.model_builder = None
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        # Create output directories
        self.model_dir = ensure_dir(config.FACIAL_MODEL_DIR)
        self.checkpoint_dir = ensure_dir(config.CHECKPOINTS_DIR / "facial")
        self.tensorboard_dir = ensure_dir(config.TENSORBOARD_DIR / "facial")
        
        print("👤 Facial Emotion Trainer Initialized")
        print(f"   Model directory: {self.model_dir}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")
    
    def load_data(self):
        """Load FER-2013 data generators."""
        print("\n📂 Loading FER-2013 dataset...")
        
        loader = FER2013DataLoader(
            image_size=config.IMAGE_SIZE,
            batch_size=32
        )
        
        # Verify dataset
        if not loader.verify_dataset():
            raise ValueError("FER-2013 dataset not found or incomplete")
        
        # Create generators
        self.train_generator, self.val_generator, self.test_generator = \
            loader.create_data_generators(validation_split=0.1)
        
        # Load class mapping
        metadata_file = config.DATA_DIR / "processed_features" / "facial_class_mapping.json"
        with open(metadata_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        print(f"\n✅ Data loaded successfully!")
        return True
    
    def calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset."""
        print("\n⚖️  Calculating class weights...")
        
        # Get class distribution from generator
        class_counts = {}
        for class_name, class_idx in self.train_generator.class_indices.items():
            class_counts[class_idx] = 0
        
        # Count samples per class
        for i in range(len(self.train_generator)):
            _, labels = self.train_generator[i]
            for label in labels:
                class_idx = np.argmax(label)
                class_counts[class_idx] += 1
        
        # Compute weights
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        class_weights = {}
        for class_idx, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[class_idx] = weight
        
        print(f"   Class distribution and weights:")
        idx_to_class = self.class_mapping['idx_to_class']
        for class_idx in sorted(class_weights.keys()):
            class_name = idx_to_class[str(class_idx)]
            count = class_counts[class_idx]
            weight = class_weights[class_idx]
            print(f"   {class_name:10s}: {count:5d} samples (weight: {weight:.3f})")
        
        return class_weights
    
    def build_model(self):
        """Build and compile the model."""
        print("\n🏗️  Building model...")
        
        input_shape = (*config.IMAGE_SIZE, 3)
        num_classes = self.class_mapping['num_classes']
        
        print(f"   Input shape: {input_shape}")
        print(f"   Number of classes: {num_classes}")
        
        # Create model
        self.model_builder = FacialEmotionModel(input_shape, num_classes)
        self.model = self.model_builder.build_model(freeze_base=True)
        self.model_builder.compile_model(learning_rate=0.001)
        
        print("✅ Model built successfully!")
    
    def create_callbacks(self, stage="transfer"):
        """Create training callbacks."""
        print(f"\n⚙️  Setting up callbacks for {stage} learning...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = []
        
        # 1. Model Checkpoint
        checkpoint_path = self.checkpoint_dir / f"facial_model_{stage}_{timestamp}.h5"
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
        patience = 10 if stage == "transfer" else 15
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print(f"   ✅ Early stopping: patience={patience}")
        
        # 3. Reduce Learning Rate
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        print(f"   ✅ Reduce LR: factor=0.5, patience=5")
        
        # 4. TensorBoard
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=str(self.tensorboard_dir / f"{stage}_{timestamp}"),
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        print(f"   ✅ TensorBoard: enabled")
        
        # 5. CSV Logger
        csv_logger = keras.callbacks.CSVLogger(
            str(self.model_dir / f"training_log_{stage}_{timestamp}.csv")
        )
        callbacks.append(csv_logger)
        print(f"   ✅ CSV Logger: enabled")
        
        return callbacks
    
    def train_transfer_learning(self, epochs=30):
        """
        Stage 1: Transfer learning with frozen base.
        
        Args:
            epochs: Number of epochs for transfer learning
        """
        print(f"\n🚀 Stage 1: Transfer Learning")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   EfficientNet base: FROZEN")
        print("="*60)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights()
        
        # Create callbacks
        callbacks = self.create_callbacks(stage="transfer")
        
        # Train
        with Timer("Transfer learning"):
            self.history = self.model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Transfer learning completed!")
    
    def train_fine_tuning(self, epochs=20):
        """
        Stage 2: Fine-tuning with partially unfrozen base.
        
        Args:
            epochs: Number of epochs for fine-tuning
        """
        print(f"\n🚀 Stage 2: Fine-Tuning")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   EfficientNet base: PARTIALLY UNFROZEN")
        print("="*60)
        
        # Unfreeze top layers
        self.model_builder.unfreeze_base_layers(num_layers_to_unfreeze=30)
        
        # Recompile with lower learning rate
        self.model_builder.compile_model(learning_rate=0.0001)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights()
        
        # Create callbacks
        callbacks = self.create_callbacks(stage="finetune")
        
        # Train
        with Timer("Fine-tuning"):
            history_finetune = self.model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Fine-tuning completed!")
        
        # Combine histories
        for key in history_finetune.history:
            if key in self.history.history:
                self.history.history[key].extend(history_finetune.history[key])
    
    def evaluate(self):
        """Evaluate model on test set."""
        print("\n📊 Evaluating model on test set...")
        
        results = self.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        print("\n📈 Test Set Results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"   {metric_name}: {value:.4f}")
        
        return results
    
    def save_final_model(self):
        """Save the final trained model."""
        print("\n💾 Saving final model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"facial_emotion_final_{timestamp}.h5"
        
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
    print("👤 Facial Expression Recognition - Training Pipeline")
    print("="*60)
    
    # Initialize trainer
    trainer = FacialEmotionTrainer()
    
    # Load data
    trainer.load_data()
    
    # Build model
    trainer.build_model()
    
    # Stage 1: Transfer Learning (frozen base)
    trainer.train_transfer_learning(epochs=30)
    
    # Stage 2: Fine-Tuning (partially unfrozen)
    trainer.train_fine_tuning(epochs=20)
    
    # Evaluate on test set
    trainer.evaluate()
    
    # Save final model
    model_path = trainer.save_final_model()
    
    print("\n" + "="*60)
    print("🎉 Facial Model Training Complete!")
    print("="*60)
    print(f"📁 Model saved: {model_path}")
    print(f"📊 View training progress:")
    print(f"   tensorboard --logdir {trainer.tensorboard_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
