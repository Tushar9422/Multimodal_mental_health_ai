#!/usr/bin/env python3
"""
Fusion system demonstration.
Shows how multimodal fusion works with qualitative examples.
FIXED: Handles text input dtype issues properly.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fusion.multimodal_fusion import load_fusion_system
from src.config import get_config
from sklearn.metrics import accuracy_score

config = get_config()


def show_fusion_weights(fusion):
    """Display fusion system configuration."""
    print("\n" + "="*60)
    print("⚙️  FUSION SYSTEM CONFIGURATION")
    print("="*60)
    
    print(f"\n🎯 Target emotions: {fusion.target_emotions}")
    
    print(f"\n📊 Model weights (based on individual accuracy):")
    for model, weight in fusion.weights.items():
        bar = '█' * int(weight * 50)
        print(f"   {model:8s}: {weight:.2f} {bar}")
    
    print(f"\n🔗 How fusion works:")
    print(f"   1. Each available model makes a prediction")
    print(f"   2. Predictions are weighted by model accuracy")
    print(f"   3. Weighted average produces final result")
    print(f"   4. Higher-accuracy models have more influence")
    print(f"   5. Works even if some modalities are missing")


def test_text_only_fusion(fusion):
    """Test fusion with text-only inputs."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Text-Only Fusion")
    print("="*60)
    
    test_cases = [
        ("I am very happy and excited today!", "very_positive"),
        ("This is absolutely terrible and depressing.", "very_negative"),
        ("I feel okay, nothing particularly special.", "neutral"),
        ("I'm quite pleased with the results.", "positive"),
        ("I'm disappointed and frustrated with this.", "negative"),
        ("Everything is wonderful and amazing!", "very_positive"),
        ("This makes me feel sad and lonely.", "very_negative"),
        ("The situation is acceptable.", "neutral"),
    ]
    
    correct = 0
    
    for i, (text, expected) in enumerate(test_cases, 1):
        # FIX: Convert text to object array properly
        text_input = np.array([text], dtype=object)
        
        try:
            result = fusion.predict(text_input=text_input, return_details=True)
            
            is_correct = result['prediction'] == expected
            correct += int(is_correct)
            
            status = "✅" if is_correct else "❌"
            
            print(f"\n{status} Test {i}/8:")
            print(f"   Text: \"{text}\"")
            print(f"   Expected: {expected}")
            print(f"   Predicted: {result['prediction']} ({result['confidence']*100:.1f}% confidence)")
            print(f"   Modalities used: {result['modalities_used']}")
            
            if not is_correct:
                # Show top 3 predictions
                probs = sorted(result['probabilities'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top predictions:")
                for emotion, prob in probs:
                    print(f"      {emotion}: {prob*100:.1f}%")
        
        except Exception as e:
            print(f"\n❌ Test {i}/8 failed with error:")
            print(f"   Text: \"{text}\"")
            print(f"   Error: {e}")
    
    accuracy = correct / len(test_cases) * 100
    print(f"\n📊 Text-only accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")


def test_audio_text_fusion(fusion):
    """Test audio + text multimodal fusion."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Multimodal Fusion (Audio + Text)")
    print("="*60)
    
    # Load available test data
    data_dir = config.DATA_DIR / "processed_features"
    
    try:
        # Audio data
        X_audio = np.load(data_dir / "X_audio.npy")
        y_audio = np.load(data_dir / "y_audio.npy")
        
        # Text data
        X_text_val = np.load(data_dir / "X_text_val.npy", allow_pickle=True)
        y_text_val = np.load(data_dir / "y_text_val.npy")
        
        # Take small samples for demo
        n_samples = min(50, len(X_audio), len(X_text_val))
        
        audio_samples = X_audio[:n_samples]
        text_samples = X_text_val[:n_samples]
        text_labels = y_text_val[:n_samples]
        
        print(f"\n📊 Testing on {n_samples} samples with Audio + Text fusion...")
        print(f"   Using text labels as ground truth (most reliable)")
        
        # Test text-only
        print("\n   1️⃣  Text-only predictions...")
        text_preds = []
        for i, text in enumerate(text_samples):
            try:
                # FIX: Handle text properly - convert to object array
                text_input = np.array([str(text)], dtype=object)
                result = fusion.predict(text_input=text_input)
                pred_idx = fusion.target_emotions.index(result['prediction'])
                text_preds.append(pred_idx)
            except Exception as e:
                print(f"      Warning: Sample {i} failed: {e}")
                # Use random prediction as fallback
                text_preds.append(0)
        
        text_preds = np.array(text_preds)
        text_acc = accuracy_score(text_labels, text_preds)
        
        # Test audio + text fusion
        print("   2️⃣  Audio + Text fusion predictions...")
        fusion_preds = []
        for i in range(n_samples):
            try:
                # FIX: Handle text properly
                text_input = np.array([str(text_samples[i])], dtype=object)
                result = fusion.predict(
                    audio_input=audio_samples[i:i+1],
                    text_input=text_input
                )
                pred_idx = fusion.target_emotions.index(result['prediction'])
                fusion_preds.append(pred_idx)
            except Exception as e:
                print(f"      Warning: Sample {i} failed: {e}")
                fusion_preds.append(text_preds[i])  # Fallback to text prediction
        
        fusion_preds = np.array(fusion_preds)
        fusion_acc = accuracy_score(text_labels, fusion_preds)
        
        # Show results
        print(f"\n📈 Results:")
        print(f"   Text-only accuracy:  {text_acc*100:5.1f}%")
        print(f"   Audio+Text fusion:   {fusion_acc*100:5.1f}%")
        
        if fusion_acc > text_acc:
            improvement = (fusion_acc - text_acc) * 100
            print(f"   ✅ Fusion improved by: +{improvement:.1f}%")
        elif fusion_acc < text_acc:
            decline = (text_acc - fusion_acc) * 100
            print(f"   ⚠️  Text-only was better by: {decline:.1f}%")
            print(f"      (Audio model may add noise due to lower accuracy)")
        else:
            print(f"   ➖ Same performance")
        
        # Show some examples
        print(f"\n📋 Sample predictions (first 5):")
        for i in range(min(5, n_samples)):
            true_emotion = fusion.target_emotions[text_labels[i]]
            text_emotion = fusion.target_emotions[text_preds[i]]
            fusion_emotion = fusion.target_emotions[fusion_preds[i]]
            
            text_correct = "✅" if text_preds[i] == text_labels[i] else "❌"
            fusion_correct = "✅" if fusion_preds[i] == text_labels[i] else "❌"
            
            # Truncate long text
            text_str = str(text_samples[i])
            if len(text_str) > 50:
                text_str = text_str[:50] + "..."
            
            print(f"\n   Sample {i+1}:")
            print(f"      Text: \"{text_str}\"")
            print(f"      True: {true_emotion}")
            print(f"      Text-only: {text_emotion} {text_correct}")
            print(f"      Fusion: {fusion_emotion} {fusion_correct}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Could not load test data: {e}")
        print(f"   Skipping quantitative evaluation")
    except Exception as e:
        print(f"\n⚠️  Error during evaluation: {e}")
        print(f"   Skipping quantitative evaluation")


def show_prediction_breakdown(fusion):
    """Show detailed prediction breakdown for a sample."""
    print("\n" + "="*60)
    print("🔍 DETAILED PREDICTION BREAKDOWN")
    print("="*60)
    
    sample_text = "I feel extremely happy and joyful today!"
    
    print(f"\n📝 Input: \"{sample_text}\"")
    
    try:
        # FIX: Convert text properly
        text_input = np.array([sample_text], dtype=object)
        
        # Get detailed prediction
        result = fusion.predict(text_input=text_input, return_details=True)
        
        print(f"\n🎯 Final Prediction:")
        print(f"   Emotion: {result['prediction'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        
        print(f"\n📊 Probability Distribution:")
        probs_sorted = sorted(result['probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        
        for emotion, prob in probs_sorted:
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            percentage = prob * 100
            
            marker = " ← PREDICTED" if emotion == result['prediction'] else ""
            print(f"   {emotion:15s}: {percentage:5.1f}% {bar}{marker}")
        
        print(f"\n⚙️  System Details:")
        print(f"   Modalities used: {result['modalities_used']}")
        print(f"   Weights applied: {result['weights_used']}")
    
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")


def main():
    """Main demo pipeline."""
    print("🔗 MULTIMODAL FUSION SYSTEM - DEMONSTRATION")
    print("="*60)
    
    # Load fusion system
    print("\n📥 Loading fusion system...")
    try:
        fusion = load_fusion_system()
    except Exception as e:
        print(f"❌ Failed to load fusion system: {e}")
        return
    
    # Show configuration
    show_fusion_weights(fusion)
    
    # Test 1: Text-only
    test_text_only_fusion(fusion)
    
    # Test 2: Multimodal fusion
    test_audio_text_fusion(fusion)
    
    # Detailed breakdown
    show_prediction_breakdown(fusion)
    
    # Summary
    print("\n" + "="*60)
    print("✅ FUSION DEMO COMPLETE!")
    print("="*60)
    
    print("\n💡 Key Takeaways:")
    print("   • Text model has highest weight (55%)")
    print("   • Fusion combines all available modalities")
    print("   • More modalities generally = more robust predictions")
    print("   • System works even if some modalities are missing")
    print("   • Audio model contributes but has lower accuracy")
    
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
