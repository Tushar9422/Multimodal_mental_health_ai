#!/usr/bin/env python3
"""
Multimodal fusion system combining audio, facial, and text models.
Uses late fusion with weighted averaging.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger

logger = get_project_logger("multimodal_fusion")
config = get_config()


class MultimodalFusionSystem:
    """
    Multimodal fusion system for mental health emotion recognition.
    
    Combines:
    - Audio emotion model (8 emotions → mapped to 5)
    - Facial emotion model (7 emotions → mapped to 5)
    - Text sentiment model (5 emotions - direct)
    
    Output: Unified 5-category emotion prediction
    """
    
    def __init__(self, audio_model_path, facial_model_path, text_model_path):
        """
        Initialize fusion system.
        
        Args:
            audio_model_path: Path to trained audio model
            facial_model_path: Path to trained facial model
            text_model_path: Path to trained text model
        """
        self.audio_model_path = Path(audio_model_path)
        self.facial_model_path = Path(facial_model_path)
        self.text_model_path = Path(text_model_path)
        
        self.audio_model = None
        self.facial_model = None
        self.text_model = None
        
        # Target emotion categories (5 unified)
        self.target_emotions = [
            'negative',
            'neutral', 
            'positive',
            'very_negative',
            'very_positive'
        ]
        
        # Fusion weights (based on individual model performance)
        # Higher accuracy = higher weight
        self.weights = {
            'audio': 0.15,   # 41.67% accuracy
            'facial': 0.30,  # 66.93% accuracy
            'text': 0.55     # 78.32% accuracy (highest weight)
        }
        
        print("🔗 Multimodal Fusion System")
        print("="*60)
        print(f"   Target emotions: {self.target_emotions}")
        print(f"   Fusion weights:")
        print(f"      Audio:  {self.weights['audio']:.2f} (41.67% acc)")
        print(f"      Facial: {self.weights['facial']:.2f} (66.93% acc)")
        print(f"      Text:   {self.weights['text']:.2f} (78.32% acc)")
    
    def load_models(self):
        """Load all three trained models."""
        print("\n📂 Loading trained models...")
        
        # Load audio model
        print(f"   Audio: {self.audio_model_path.name}")
        self.audio_model = keras.models.load_model(self.audio_model_path)
        print("   ✅ Audio model loaded")
        
        # Load facial model
        print(f"   Facial: {self.facial_model_path.name}")
        self.facial_model = keras.models.load_model(self.facial_model_path)
        print("   ✅ Facial model loaded")
        
        # Load text model (rebuild to avoid serialization issues)
        print(f"   Text: {self.text_model_path.name}")
        self._load_text_model()
        print("   ✅ Text model loaded")
        
        print("\n✅ All models loaded successfully!")
    
    def _load_text_model(self):
        """Load text model (handles custom vectorization)."""
        from src.models.text_sentiment_model import TextSentimentModel
        
        # Load metadata
        metadata_file = config.DATA_DIR / "processed_features" / "text_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load training data for vectorizer
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
        
        # Load weights
        model_builder.model.load_weights(str(self.text_model_path))
        self.text_model = model_builder.model
    
    def map_audio_to_unified(self, audio_probs):
        """
        Map 8 RAVDESS emotions to 5 unified categories.
        
        RAVDESS order: neutral, calm, happy, sad, angry, fearful, disgust, surprised
        Target order: negative, neutral, positive, very_negative, very_positive
        """
        batch_size = audio_probs.shape[0]
        
        # Mapping matrix (8 RAVDESS → 5 unified)
        # Each row sums to 1.0
        mapping = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0],  # neutral → neutral
            [0.0, 1.0, 0.0, 0.0, 0.0],  # calm → neutral
            [0.0, 0.0, 0.6, 0.0, 0.4],  # happy → positive/very_positive
            [0.7, 0.0, 0.0, 0.3, 0.0],  # sad → negative/very_negative
            [0.5, 0.0, 0.0, 0.5, 0.0],  # angry → negative/very_negative
            [0.6, 0.0, 0.0, 0.4, 0.0],  # fearful → negative/very_negative
            [0.4, 0.0, 0.0, 0.6, 0.0],  # disgust → very_negative/negative
            [0.0, 0.0, 0.7, 0.0, 0.3],  # surprised → positive/very_positive
        ])
        
        # Apply mapping: (batch, 8) @ (8, 5) = (batch, 5)
        unified_probs = np.dot(audio_probs, mapping)
        
        # Normalize (should already sum to 1, but ensure)
        unified_probs = unified_probs / unified_probs.sum(axis=-1, keepdims=True)
        
        return unified_probs
    
    def map_facial_to_unified(self, facial_probs):
        """
        Map 7 FER-2013 emotions to 5 unified categories.
        
        FER order: angry, disgust, fear, happy, neutral, sad, surprise
        Target order: negative, neutral, positive, very_negative, very_positive
        """
        # Mapping matrix (7 FER → 5 unified)
        mapping = np.array([
            [0.5, 0.0, 0.0, 0.5, 0.0],  # angry → negative/very_negative
            [0.4, 0.0, 0.0, 0.6, 0.0],  # disgust → very_negative/negative
            [0.6, 0.0, 0.0, 0.4, 0.0],  # fear → negative/very_negative
            [0.0, 0.0, 0.6, 0.0, 0.4],  # happy → positive/very_positive
            [0.0, 1.0, 0.0, 0.0, 0.0],  # neutral → neutral
            [0.7, 0.0, 0.0, 0.3, 0.0],  # sad → negative/very_negative
            [0.0, 0.0, 0.7, 0.0, 0.3],  # surprise → positive/very_positive
        ])
        
        # Apply mapping
        unified_probs = np.dot(facial_probs, mapping)
        
        # Normalize
        unified_probs = unified_probs / unified_probs.sum(axis=-1, keepdims=True)
        
        return unified_probs
    
    def fuse_predictions(self, audio_input=None, facial_input=None, text_input=None):
        """
        Fuse predictions from available modalities.
        
        Args:
            audio_input: Audio features (MFCCs) - shape (batch, time, features)
            facial_input: Facial images - shape (batch, 224, 224, 3)
            text_input: Text strings - shape (batch,) or (batch, 1)
            
        Returns:
            Fused probability distribution over 5 emotions - shape (batch, 5)
        """
        predictions = []
        weights_used = []
        modalities_used = []
        
        # Get audio prediction
        if audio_input is not None and self.audio_model is not None:
            audio_probs = self.audio_model.predict(audio_input, verbose=0)
            unified_audio = self.map_audio_to_unified(audio_probs)
            predictions.append(unified_audio)
            weights_used.append(self.weights['audio'])
            modalities_used.append('audio')
        
        # Get facial prediction
        if facial_input is not None and self.facial_model is not None:
            facial_probs = self.facial_model.predict(facial_input, verbose=0)
            unified_facial = self.map_facial_to_unified(facial_probs)
            predictions.append(unified_facial)
            weights_used.append(self.weights['facial'])
            modalities_used.append('facial')
        
        # Get text prediction (already in unified format)
        if text_input is not None and self.text_model is not None:
            # Ensure text input is 1D array of strings
            if len(text_input.shape) > 1:
                text_input = text_input.squeeze()
            text_probs = self.text_model.predict(text_input, verbose=0)
            predictions.append(text_probs)
            weights_used.append(self.weights['text'])
            modalities_used.append('text')
        
        if len(predictions) == 0:
            raise ValueError("At least one input modality required")
        
        # Normalize weights to sum to 1.0
        weights_array = np.array(weights_used)
        weights_array = weights_array / weights_array.sum()
        
        # Weighted average fusion
        fused_probs = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights_array):
            fused_probs += weight * pred
        
        return fused_probs, modalities_used, weights_array
    
    def predict(self, audio_input=None, facial_input=None, text_input=None, return_details=False):
        """
        Make unified prediction from available modalities.
        
        Returns:
            If return_details=False: Dictionary with prediction and confidence
            If return_details=True: Dictionary with full details
        """
        # Get fused probabilities
        fused_probs, modalities_used, weights_used = self.fuse_predictions(
            audio_input, facial_input, text_input
        )
        
        # Get prediction (for each sample in batch)
        pred_indices = np.argmax(fused_probs, axis=-1)
        confidences = np.max(fused_probs, axis=-1)
        
        # For single sample, return scalar values
        if len(pred_indices) == 1:
            pred_idx = int(pred_indices[0])
            pred_emotion = self.target_emotions[pred_idx]
            confidence = float(confidences[0])
            
            result = {
                'prediction': pred_emotion,
                'confidence': confidence,
                'probabilities': {
                    emotion: float(fused_probs[0, i])
                    for i, emotion in enumerate(self.target_emotions)
                }
            }
            
            if return_details:
                result['modalities_used'] = modalities_used
                result['weights_used'] = {
                    mod: float(w) for mod, w in zip(modalities_used, weights_used)
                }
            
            return result
        
        # For batch, return arrays
        predictions = [self.target_emotions[idx] for idx in pred_indices]
        
        result = {
            'predictions': predictions,
            'confidences': confidences.tolist(),
            'probabilities': fused_probs
        }
        
        if return_details:
            result['modalities_used'] = modalities_used
            result['weights_used'] = {
                mod: float(w) for mod, w in zip(modalities_used, weights_used)
            }
        
        return result


def load_fusion_system():
    """Load complete fusion system with all trained models."""
    print("🔗 Loading Multimodal Fusion System")
    print("="*60)
    
    # Find latest models
    audio_models = list(config.AUDIO_MODEL_DIR.glob("audio_emotion_final_*.h5"))
    facial_models = list(config.FACIAL_MODEL_DIR.glob("facial_emotion_final_*.h5"))
    text_models = list(config.TEXT_MODEL_DIR.glob("text_sentiment_final_*.h5"))
    
    if not audio_models:
        raise FileNotFoundError(f"No audio model found in {config.AUDIO_MODEL_DIR}")
    if not facial_models:
        raise FileNotFoundError(f"No facial model found in {config.FACIAL_MODEL_DIR}")
    if not text_models:
        raise FileNotFoundError(f"No text model found in {config.TEXT_MODEL_DIR}")
    
    # Get latest of each
    audio_model = max(audio_models, key=lambda p: p.stat().st_mtime)
    facial_model = max(facial_models, key=lambda p: p.stat().st_mtime)
    text_model = max(text_models, key=lambda p: p.stat().st_mtime)
    
    print(f"\n📁 Models to load:")
    print(f"   Audio:  {audio_model.name}")
    print(f"   Facial: {facial_model.name}")
    print(f"   Text:   {text_model.name}")
    
    # Create fusion system
    fusion = MultimodalFusionSystem(audio_model, facial_model, text_model)
    fusion.load_models()
    
    print("\n" + "="*60)
    print("✅ Fusion system ready for inference!")
    print("="*60)
    
    return fusion


if __name__ == "__main__":
    # Test loading fusion system
    fusion = load_fusion_system()
    
    print("\n🎉 Fusion system loaded successfully!")
    print(f"   Target emotions: {fusion.target_emotions}")
    print(f"   Ready for multimodal predictions!")
