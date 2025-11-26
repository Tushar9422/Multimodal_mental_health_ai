"""
Evaluation and visualization for audio emotion recognition model.

Provides:
- Confusion matrix
- Classification report
- Training history visualization
- Per-class accuracy analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import ensure_dir

config = get_config()


class AudioModelEvaluator:
    """Evaluator for audio emotion recognition model."""
    
    def __init__(self, model_path):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.emotion_labels = None
        
        # Create output directory for plots
        self.output_dir = ensure_dir(config.DATA_DIR / "evaluation" / "audio")
        
        print(f"📊 Audio Model Evaluator")
        print(f"   Model: {model_path}")
        print(f"   Output: {self.output_dir}")
    
    def load_model(self):
        """Load trained model."""
        print("\n📂 Loading model...")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
    
    def load_data(self):
        """Load test data."""
        print("\n📂 Loading test data...")
        
        data_dir = config.DATA_DIR / "processed_features"
        
        # Load features and labels
        X = np.load(data_dir / "X_audio.npy")
        y = np.load(data_dir / "y_audio.npy")
        
        # Load emotion mapping
        with open(data_dir / "emotion_mapping.json", 'r') as f:
            emotion_mapping = json.load(f)
        
        self.emotion_labels = emotion_mapping['emotions']
        
        # Convert to one-hot
        y_categorical = keras.utils.to_categorical(y, len(self.emotion_labels))
        
        print(f"✅ Data loaded!")
        print(f"   Samples: {len(X)}")
        print(f"   Emotions: {self.emotion_labels}")
        
        return X, y_categorical, y
    
    def evaluate_model(self, X, y):
        """Evaluate model and get predictions."""
        print("\n🔍 Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y, axis=1)
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        print("\n📊 Creating confusion matrix...")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot absolute counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Plot normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            ax=ax2
        )
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {output_path}")
        
        plt.close()
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report."""
        print("\n📋 Classification Report:")
        print("="*60)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.emotion_labels,
            digits=4
        )
        
        print(report)
        
        # Save to file
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")
    
    def plot_training_history(self, history_path):
        """Plot training history curves."""
        print("\n📈 Plotting training history...")
        
        # Load history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training')
        axes[0, 0].plot(history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training')
        axes[0, 1].plot(history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training')
            axes[1, 0].plot(history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training')
            axes[1, 1].plot(history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "training_history.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved: {output_path}")
        
        plt.close()
    
    def analyze_per_class_accuracy(self, y_true, y_pred):
        """Analyze accuracy for each emotion class."""
        print("\n🎯 Per-Class Accuracy Analysis:")
        print("="*60)
        
        for idx, emotion in enumerate(self.emotion_labels):
            # Get samples for this emotion
            mask = (y_true == idx)
            if mask.sum() == 0:
                continue
            
            # Calculate accuracy
            correct = (y_pred[mask] == y_true[mask]).sum()
            total = mask.sum()
            accuracy = correct / total * 100
            
            print(f"{emotion:12s}: {accuracy:6.2f}% ({correct}/{total})")


def main():
    """Main evaluation pipeline."""
    print("📊 Audio Emotion Recognition - Evaluation")
    print("="*60)
    
    # Find latest model
    model_dir = config.AUDIO_MODEL_DIR
    model_files = list(model_dir.glob("audio_emotion_final_*.h5"))
    
    if not model_files:
        print("❌ No trained model found!")
        return False
    
    # Get latest model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using model: {latest_model}")
    
    # Find corresponding history file
    timestamp = latest_model.stem.split('_')[-1]
    history_file = model_dir / f"training_history_{timestamp}.json"
    
    # Initialize evaluator
    evaluator = AudioModelEvaluator(latest_model)
    
    # Load model and data
    evaluator.load_model()
    X, y_categorical, y_original = evaluator.load_data()
    
    # Evaluate
    y_true, y_pred, y_pred_proba = evaluator.evaluate_model(X, y_categorical)
    
    # Visualizations
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.print_classification_report(y_true, y_pred)
    evaluator.analyze_per_class_accuracy(y_true, y_pred)
    
    if history_file.exists():
        evaluator.plot_training_history(history_file)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print(f"📁 Results saved to: {evaluator.output_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
