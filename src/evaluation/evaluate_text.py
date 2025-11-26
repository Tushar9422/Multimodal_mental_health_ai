#!/usr/bin/env python3
"""
Evaluation and visualization for text sentiment analysis model.
Fixed for BiLSTM model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

from tensorflow.keras.layers import TextVectorization  



import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import ensure_dir

config = get_config()


class TextModelEvaluator:
    """Evaluator for text sentiment analysis model."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.categories = None
        
        self.output_dir = ensure_dir(config.DATA_DIR / "evaluation" / "text")
        
        print(f"📊 Text Model Evaluator")
        print(f"   Model: {model_path}")
        print(f"   Output: {self.output_dir}")
    
    def load_model(self):
        """
        Load trained model.
        Handles custom TextVectorization preprocessing function.
        """
        print("\n📂 Loading model...")
        
        # Import TextVectorization
        
        
        # Recreate the custom standardization function
        # Must match exactly what was used during training
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
            cleaned = tf.strings.regex_replace(
                stripped_html,
                r'[^a-z0-9\s\']',
                ''
            )
            return cleaned
        
        # Try to load with custom objects
        try:
            self.model = keras.models.load_model(
                str(self.model_path),
                custom_objects={
                    'custom_standardization': custom_standardization,
                    'TextVectorization': TextVectorization
                }
            )
            print("✅ Model loaded with custom objects")
        except Exception as e:
            print(f"⚠️  Could not load with custom objects: {e}")
            print("   Trying alternative loading method...")
            
            # Alternative: Load architecture and weights separately
            # This is more complex but works around serialization issues
            from src.models.text_sentiment_model import TextSentimentModel
            
            # Load metadata
            metadata_file = config.DATA_DIR / "processed_features" / "text_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load training data to adapt vectorizer
            data_dir = config.DATA_DIR / "processed_features"
            X_train = np.load(data_dir / "X_text_train.npy", allow_pickle=True)
            
            # Rebuild model
            model_builder = TextSentimentModel(
                num_classes=metadata['num_classes'],
                max_tokens=10000,
                max_length=metadata['max_length']
            )
            model_builder.create_vectorizer(X_train)
            model_builder.build_model()
            
            # Load weights only
            model_builder.model.load_weights(str(self.model_path))
            self.model = model_builder.model
            
            print("✅ Model loaded with weight restoration")
        
        # Load metadata for categories
        metadata_file = config.DATA_DIR / "processed_features" / "text_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.categories = metadata['categories']
        
        print(f"   Categories: {self.categories}")

    
    def load_test_data(self):
        """Load validation data for evaluation."""
        print("\n📂 Loading validation data...")
        
        data_dir = config.DATA_DIR / "processed_features"
        
        X_val = np.load(data_dir / "X_text_val.npy", allow_pickle=True)
        y_val = np.load(data_dir / "y_text_val.npy")
        
        print(f"✅ Validation data loaded: {len(X_val)} samples")
        
        return X_val, y_val
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate model and get predictions."""
        print("\n🔍 Evaluating model...")
        
        # Create dataset (no reshaping needed - keep as 1D)
        print("   Creating dataset...")
        dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        dataset = dataset.batch(32)
        
        # Get predictions
        print("   Making predictions...")
        y_pred_proba = self.model.predict(dataset, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = y_val
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"\n✅ Validation Accuracy: {accuracy*100:.2f}%")
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        print("\n📊 Creating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories, ax=ax2)
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
            target_names=self.categories,
            digits=4
        )
        
        print(report)
        
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")


def main():
    """Main evaluation pipeline."""
    print("📊 Text Sentiment Analysis - Evaluation")
    print("="*60)
    
    # Find latest model
    model_dir = config.TEXT_MODEL_DIR
    model_files = list(model_dir.glob("text_sentiment_final_*.h5"))
    
    if not model_files:
        print("❌ No trained model found!")
        return False
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using model: {latest_model}")
    
    # Initialize evaluator
    evaluator = TextModelEvaluator(latest_model)
    
    # Load model and data
    evaluator.load_model()
    X_val, y_val = evaluator.load_test_data()
    
    # Evaluate
    y_true, y_pred, y_pred_proba = evaluator.evaluate_model(X_val, y_val)
    
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
