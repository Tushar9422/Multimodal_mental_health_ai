#!/usr/bin/env python3
"""
Evaluation and visualization for facial expression recognition model.
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

from src.preprocessing.image_processor import FER2013DataLoader
from src.config import get_config
from src.utils import ensure_dir

config = get_config()


class FacialModelEvaluator:
    """Evaluator for facial emotion recognition model."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.emotion_labels = None
        
        self.output_dir = ensure_dir(config.DATA_DIR / "evaluation" / "facial")
        
        print(f"📊 Facial Model Evaluator")
        print(f"   Model: {model_path}")
        print(f"   Output: {self.output_dir}")
    
    def load_model(self):
        """Load trained model."""
        print("\n📂 Loading model...")
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
    
    def load_test_data(self):
        """Load test data."""
        print("\n📂 Loading test data...")
        
        loader = FER2013DataLoader(
            image_size=config.IMAGE_SIZE,
            batch_size=32
        )
        
        # Load class mapping
        metadata_file = config.DATA_DIR / "processed_features" / "facial_class_mapping.json"
        with open(metadata_file, 'r') as f:
            class_mapping = json.load(f)
        
        self.emotion_labels = [class_mapping['idx_to_class'][str(i)] 
                              for i in range(class_mapping['num_classes'])]
        
        # Create test generator
        _, _, test_generator = loader.create_data_generators()
        
        print(f"✅ Test data loaded: {test_generator.samples} samples")
        
        return test_generator
    
    def evaluate_model(self, test_generator):
        """Evaluate model and get predictions."""
        print("\n🔍 Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        print("\n📊 Creating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
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
        
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")


def main():
    """Main evaluation pipeline."""
    print("📊 Facial Expression Recognition - Evaluation")
    print("="*60)
    
    # Find latest model
    model_dir = config.FACIAL_MODEL_DIR
    model_files = list(model_dir.glob("facial_emotion_final_*.h5"))
    
    if not model_files:
        print("❌ No trained model found!")
        return False
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using model: {latest_model}")
    
    # Initialize evaluator
    evaluator = FacialModelEvaluator(latest_model)
    
    # Load model and data
    evaluator.load_model()
    test_generator = evaluator.load_test_data()
    
    # Evaluate
    y_true, y_pred, y_pred_proba = evaluator.evaluate_model(test_generator)
    
    # Visualizations
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.print_classification_report(y_true, y_pred)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print(f"📁 Results saved to: {evaluator.output_dir}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
