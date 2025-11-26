#!/usr/bin/env python3
"""
Real-time inference engine for multimodal emotion recognition.
Handles all modalities with cached models for fast inference.
"""

import numpy as np
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fusion.multimodal_fusion import load_fusion_system
from src.inference.audio_processor import AudioProcessor
from src.inference.image_processor import ImageProcessor
from src.config import get_config

config = get_config()


class RealtimeInferenceEngine:
    """
    Real-time inference engine for multimodal emotion recognition.
    
    Features:
    - Models loaded once (cached)
    - Fast inference (<1 second)
    - Handles any combination of modalities
    - Returns detailed predictions with confidence
    """
    
    def __init__(self, load_models=True):
        """
        Initialize inference engine.
        
        Args:
            load_models: Whether to load models on initialization
        """
        print("🚀 Initializing Real-Time Inference Engine")
        print("="*60)
        
        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        
        # Load fusion system (includes all models)
        self.fusion_system = None
        if load_models:
            self.load_models()
        
        print("\n" + "="*60)
        print("✅ Inference Engine Ready!")
        print("="*60)
    
    def load_models(self):
        """Load all models into memory."""
        print("\n📥 Loading models...")
        start_time = time.time()
        
        self.fusion_system = load_fusion_system()
        
        load_time = time.time() - start_time
        print(f"\n✅ All models loaded in {load_time:.2f} seconds")
    
    def predict_audio(self, audio_input):
        """
        Predict emotion from audio.
        
        Args:
            audio_input: Audio file path or file-like object
            
        Returns:
            Dictionary with prediction results
        """
        if self.fusion_system is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("\n🎵 Processing audio...")
        start_time = time.time()
        
        try:
            # Process audio
            mfccs = self.audio_processor.process(audio_input)
            
            # Get audio info
            audio_info = self.audio_processor.get_audio_info(audio_input)
            
            # Predict
            result = self.fusion_system.predict(
                audio_input=mfccs,
                return_details=True
            )
            
            process_time = time.time() - start_time
            
            result['processing_time'] = process_time
            result['audio_info'] = audio_info
            result['input_shape'] = mfccs.shape
            
            print(f"✅ Audio processed in {process_time:.3f}s")
            
            return result
        
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return {'error': str(e)}
    
    def predict_image(self, image_input, detect_face=False):
        """
        Predict emotion from facial image.
        
        Args:
            image_input: Image file path or file-like object
            detect_face: Whether to detect and crop face first
            
        Returns:
            Dictionary with prediction results
        """
        if self.fusion_system is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("\n📸 Processing image...")
        start_time = time.time()
        
        try:
            # Optional face detection
            if detect_face:
                image_input = self.image_processor.detect_face(image_input)
            
            # Process image
            img_array = self.image_processor.process(image_input)
            
            # Get image info
            image_info = self.image_processor.get_image_info(image_input)
            
            # Predict
            result = self.fusion_system.predict(
                facial_input=img_array,
                return_details=True
            )
            
            process_time = time.time() - start_time
            
            result['processing_time'] = process_time
            result['image_info'] = image_info
            result['input_shape'] = img_array.shape
            result['face_detected'] = detect_face
            
            print(f"✅ Image processed in {process_time:.3f}s")
            
            return result
        
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            return {'error': str(e)}
    
    def predict_text(self, text):
        """
        Predict emotion from text.
        
        Args:
            text: Text string
            
        Returns:
            Dictionary with prediction results
        """
        if self.fusion_system is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print(f"\n💬 Processing text: \"{text[:50]}...\"")
        start_time = time.time()
        
        try:
            # Convert to proper format
            text_input = np.array([text], dtype=object)
            
            # Predict
            result = self.fusion_system.predict(
                text_input=text_input,
                return_details=True
            )
            
            process_time = time.time() - start_time
            
            result['processing_time'] = process_time
            result['text_length'] = len(text)
            result['text_preview'] = text[:100]
            
            print(f"✅ Text processed in {process_time:.3f}s")
            
            return result
        
        except Exception as e:
            print(f"❌ Text processing failed: {e}")
            return {'error': str(e)}
    
    def predict_multimodal(self, audio=None, image=None, text=None, 
                          detect_face=False):
        """
        Predict emotion from multiple modalities.
        
        Args:
            audio: Audio file (optional)
            image: Image file (optional)
            text: Text string (optional)
            detect_face: Whether to detect face in image
            
        Returns:
            Dictionary with fused prediction results
        """
        if self.fusion_system is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        print("\n🔗 Processing multimodal input...")
        start_time = time.time()
        
        try:
            # Process each modality
            audio_input = None
            image_input = None
            text_input = None
            
            modalities_provided = []
            
            if audio is not None:
                audio_input = self.audio_processor.process(audio)
                modalities_provided.append('audio')
            
            if image is not None:
                if detect_face:
                    image = self.image_processor.detect_face(image)
                image_input = self.image_processor.process(image)
                modalities_provided.append('facial')
            
            if text is not None:
                text_input = np.array([text], dtype=object)
                modalities_provided.append('text')
            
            if not modalities_provided:
                raise ValueError("At least one modality required")
            
            # Fused prediction
            result = self.fusion_system.predict(
                audio_input=audio_input,
                facial_input=image_input,
                text_input=text_input,
                return_details=True
            )
            
            process_time = time.time() - start_time
            
            result['processing_time'] = process_time
            result['modalities_provided'] = modalities_provided
            
            print(f"✅ Multimodal prediction in {process_time:.3f}s")
            print(f"   Modalities used: {modalities_provided}")
            
            return result
        
        except Exception as e:
            print(f"❌ Multimodal processing failed: {e}")
            return {'error': str(e)}
    
    def get_system_info(self):
        """Get system configuration and status."""
        return {
            'models_loaded': self.fusion_system is not None,
            'fusion_weights': self.fusion_system.weights if self.fusion_system else None,
            'target_emotions': self.fusion_system.target_emotions if self.fusion_system else None,
            'audio_config': {
                'sample_rate': self.audio_processor.sample_rate,
                'n_mfcc': self.audio_processor.n_mfcc,
                'max_length': self.audio_processor.max_length
            },
            'image_config': {
                'target_size': self.image_processor.target_size
            }
        }


# Global instance for Streamlit caching
_engine_instance = None

def get_inference_engine():
    """Get or create inference engine (singleton pattern)."""
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = RealtimeInferenceEngine(load_models=True)
    
    return _engine_instance


if __name__ == "__main__":
    # Test inference engine
    print("🧪 Testing Real-Time Inference Engine")
    print("="*60)
    
    # Initialize
    engine = RealtimeInferenceEngine()
    
    # Show system info
    info = engine.get_system_info()
    print(f"\n📊 System Info:")
    print(f"   Models loaded: {info['models_loaded']}")
    print(f"   Target emotions: {info['target_emotions']}")
    
    # Test text prediction
    print("\n" + "="*60)
    print("TEST 1: Text-only prediction")
    print("="*60)
    
    result = engine.predict_text("I am very happy today!")
    print(f"\n📊 Result:")
    print(f"   Emotion: {result['prediction']}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Processing time: {result['processing_time']:.3f}s")
    
    print("\n" + "="*60)
    print("✅ Inference Engine Test Complete!")
    print("="*60)
