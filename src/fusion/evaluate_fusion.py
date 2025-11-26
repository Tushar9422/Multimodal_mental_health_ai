#!/usr/bin/env python3
"""
Evaluation script for multimodal fusion system.
Tests audio + text fusion (works without facial .npy files).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from pathlib import Path
import json
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fusion.multimodal_fusion import load_fusion_system
from src.config import get_config
from src.utils import ensure_dir

config = get_config()


class FusionSystemEvaluator:
    """
    Evaluator for multimodal fusion system.
    Tests fusion on audio + text modalities.
    """
    
    def __init__(self, fusion_system):
        self.fusion = fusion_system
        self.output_dir = ensure_dir(config.DATA_DIR / "evaluation" / "fusion")
        
        print("📊 Fusion System Evaluator")
        print(f"   Output directory: {self.output_dir}")
    
    def load_test_data(self, num_samples=500):
        """
        Load test data from audio and text modalities.
        
        Args:
            num_samples: Number of samples to test (limited for speed)
        """
        print(f"\n📂 Loading test data (up to {num_samples} samples)...")
        
        data_dir = config.DATA_DIR / "processed_features"
        
        # ============================================
        # Load AUDIO data
        # ============================================
        print("   🎵 Loading audio data...")
        X_audio_full = np.load(data_dir / "X_audio.npy")
        y_audio_full = np.load(data_dir / "y_audio.npy")
        
        # Take last 20% as validation
        val_size = int(len(X_audio_full) * 0.2)
        X_audio_val = X_audio_full[-val_size:]
        y_audio_val = y_audio_full[-val_size:]
        
        num_audio = min(num_samples, len(X_audio_val))
        X_audio_test = X_audio_val[:num_audio]
        y_audio_test = y_audio_val[:num_audio]
        
        print(f"   ✅ Audio: {len(X_audio_test)} samples")
        
        # ============================================
        # Load TEXT data
        # ============================================
        print("   💬 Loading text data...")
        X_text_val = np.load(data_dir / "X_text_val.npy", allow_pickle=True)
        y_text_val = np.load(data_dir / "y_text_val.npy")
        
        num_text = min(num_samples, len(X_text_val))
        X_text_test = X_text_val[:num_text]
        y_text_test = y_text_val[:num_text]
        
        print(f"   ✅ Text: {len(X_text_test)} samples")
        
        print(f"\n✅ Test data loaded successfully")
        
        return {
            'audio': (X_audio_test, y_audio_test),
            'text': (X_text_test, y_text_test)
        }
    
    def evaluate_audio_model(self, X_audio, y_audio):
        """Evaluate audio model individually."""
        print("\n🎵 Evaluating Audio Model...")
        
        # Get predictions
        audio_probs = self.fusion.audio_model.predict(X_audio, verbose=0)
        audio_mapped = self.fusion.map_audio_to_unified(audio_probs)
        audio_preds = np.argmax(audio_mapped, axis=-1)
        
        # Map audio labels to unified (RAVDESS → unified)
        audio_label_mapping = {
            0: 1,  # neutral → neutral
            1: 1,  # calm → neutral
            2: 2,  # happy → positive
            3: 0,  # sad → negative
            4: 3,  # angry → very_negative
            5: 0,  # fearful → negative
            6: 3,  # disgust → very_negative
            7: 2   # surprised → positive
        }
        y_audio_mapped = np.array([audio_label_mapping.get(y, 1) for y in y_audio])
        
        # Calculate accuracy
        audio_acc = accuracy_score(y_audio_mapped, audio_preds)
        
        print(f"   ✅ Audio accuracy: {audio_acc*100:.2f}%")
        
        return {
            'accuracy': audio_acc,
            'predictions': audio_preds,
            'labels': y_audio_mapped
        }
    
    def evaluate_text_model(self, X_text, y_text):
        """Evaluate text model individually."""
        print("\n💬 Evaluating Text Model...")
        
        # Get predictions
        text_probs = self.fusion.text_model.predict(X_text, verbose=0)
        text_preds = np.argmax(text_probs, axis=-1)
        
        # Calculate accuracy
        text_acc = accuracy_score(y_text, text_preds)
        
        print(f"   ✅ Text accuracy: {text_acc*100:.2f}%")
        
        return {
            'accuracy': text_acc,
            'predictions': text_preds,
            'labels': y_text
        }
    
    def evaluate_fusion(self, test_data, batch_size=32):
        """
        Evaluate fusion system on audio + text data.
        
        Uses text labels as ground truth (most reliable).
        """
        print("\n🔗 Evaluating Fusion System (Audio + Text)...")
        
        X_audio, _ = test_data['audio']
        X_text, y_text = test_data['text']
        
        # Take minimum length
        min_len = min(len(X_audio), len(X_text))
        X_audio = X_audio[:min_len]
        X_text = X_text[:min_len]
        y_true = y_text[:min_len]
        
        print(f"   Testing on {min_len} samples with both modalities")
        
        # Get fusion predictions
        all_preds = []
        all_probs = []
        
        for i in tqdm(range(0, min_len, batch_size), desc="   Fusion inference"):
            end_idx = min(i + batch_size, min_len)
            
            audio_batch = X_audio[i:end_idx]
            text_batch = X_text[i:end_idx]
            
            # Get fusion prediction
            fused_probs, _, _ = self.fusion.fuse_predictions(
                audio_input=audio_batch,
                text_input=text_batch
            )
            
            batch_preds = np.argmax(fused_probs, axis=-1)
            all_preds.extend(batch_preds)
            all_probs.append(fused_probs)
        
        y_pred = np.array(all_preds)
        y_probs = np.vstack(all_probs)
        
        # Calculate accuracy
        fusion_acc = accuracy_score(y_true, y_pred)
        
        print(f"   ✅ Fusion accuracy: {fusion_acc*100:.2f}%")
        
        return {
            'accuracy': fusion_acc,
            'predictions': y_pred,
            'probabilities': y_probs,
            'labels': y_true
        }
    
    def compare_models(self, audio_results, text_results, fusion_results):
        """Compare all approaches."""
        print("\n📊 Model Comparison:")
        print("="*60)
        
        accuracies = {
            'Audio': audio_results['accuracy'],
            'Text': text_results['accuracy'],
            'Fusion (Audio+Text)': fusion_results['accuracy']
        }
        
        for model, acc in accuracies.items():
            bar = '█' * int(acc * 50)
            print(f"   {model:20s}: {acc*100:5.2f}% {bar}")
        
        print("="*60)
        
        # Calculate improvements
        best_individual = max(audio_results['accuracy'], text_results['accuracy'])
        
        if fusion_results['accuracy'] > best_individual:
            improvement = (fusion_results['accuracy'] - best_individual) * 100
            print(f"\n✅ Fusion improvement over best individual: +{improvement:.2f}%")
        else:
            diff = (best_individual - fusion_results['accuracy']) * 100
            print(f"\n⚠️  Best individual was better by: {diff:.2f}%")
        
        # Text vs Fusion comparison (most important)
        text_vs_fusion = (fusion_results['accuracy'] - text_results['accuracy']) * 100
        if text_vs_fusion > 0:
            print(f"✅ Fusion improved over text by: +{text_vs_fusion:.2f}%")
        else:
            print(f"⚠️  Text alone was better by: {abs(text_vs_fusion):.2f}%")
            print("   (Audio model may be adding noise due to lower accuracy)")
        
        return accuracies
    
    def plot_comparison(self, accuracies):
        """Plot model comparison bar chart."""
        print("\n📊 Creating comparison plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(accuracies.keys())
        accs = [accuracies[m] * 100 for m in models]
        
        colors = ['#ff6b6b', '#4ecdc4', '#f9ca24']
        bars = ax.bar(models, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Multimodal Fusion vs Individual Models', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {output_path}")
        
        plt.close()
    
    def plot_fusion_confusion_matrix(self, fusion_results):
        """Plot confusion matrix for fusion system."""
        print("\n📊 Creating fusion confusion matrix...")
        
        y_true = fusion_results['labels']
        y_pred = fusion_results['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.fusion.target_emotions,
                   yticklabels=self.fusion.target_emotions, ax=ax1,
                   cbar_kws={'label': 'Count'})
        ax1.set_title('Fusion System - Confusion Matrix (Counts)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.fusion.target_emotions,
                   yticklabels=self.fusion.target_emotions, ax=ax2,
                   cbar_kws={'label': 'Proportion'})
        ax2.set_title('Fusion System - Confusion Matrix (Normalized)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "fusion_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {output_path}")
        
        plt.close()
    
    def print_classification_report(self, fusion_results):
        """Print detailed classification report for fusion."""
        print("\n📋 Fusion Classification Report:")
        print("="*60)
        
        y_true = fusion_results['labels']
        y_pred = fusion_results['predictions']
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.fusion.target_emotions,
            digits=4,
            zero_division=0
        )
        
        print(report)
        
        # Save report
        report_path = self.output_dir / "fusion_classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("MULTIMODAL FUSION SYSTEM - CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("Modalities: Audio + Text\n")
            f.write(f"Fusion Strategy: Weighted averaging\n")
            f.write(f"Weights: Audio={self.fusion.weights['audio']:.2f}, ")
            f.write(f"Text={self.fusion.weights['text']:.2f}\n\n")
            f.write(report)
        
        print(f"\n✅ Report saved: {report_path}")


def main():
    """Main evaluation pipeline."""
    print("🔗 MULTIMODAL FUSION SYSTEM - EVALUATION")
    print("="*60)
    
    # Load fusion system
    fusion = load_fusion_system()
    
    # Create evaluator
    evaluator = FusionSystemEvaluator(fusion)
    
    # Load test data
    test_data = evaluator.load_test_data(num_samples=500)
    
    # Evaluate individual models
    audio_results = evaluator.evaluate_audio_model(*test_data['audio'])
    text_results = evaluator.evaluate_text_model(*test_data['text'])
    
    # Evaluate fusion system
    fusion_results = evaluator.evaluate_fusion(test_data)
    
    # Compare models
    accuracies = evaluator.compare_models(audio_results, text_results, fusion_results)
    
    # Create visualizations
    evaluator.plot_comparison(accuracies)
    evaluator.plot_fusion_confusion_matrix(fusion_results)
    evaluator.print_classification_report(fusion_results)
    
    print("\n" + "="*60)
    print("✅ FUSION EVALUATION COMPLETE!")
    print("="*60)
    print(f"📁 Results saved to: {evaluator.output_dir}")
    print("\nKey Findings:")
    print(f"   • Audio model: {audio_results['accuracy']*100:.2f}%")
    print(f"   • Text model: {text_results['accuracy']*100:.2f}%")
    print(f"   • Fusion system: {fusion_results['accuracy']*100:.2f}%")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
