#!/usr/bin/env python3
"""Comprehensive WSL2 setup test."""

import sys
import time
import numpy as np

def test_tensorflow_gpu():
    """Test TensorFlow GPU functionality."""
    print("🧪 Testing TensorFlow GPU Performance")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        
        # Basic info
        print(f"TensorFlow: {tf.__version__}")
        print(f"Eager Execution: {tf.executing_eagerly()}")
        
        # GPU test
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU Devices: {len(gpus)}")
        
        if gpus:
            # Performance test
            print("Running GPU performance test...")
            
            with tf.device('/GPU:0'):
                # Matrix multiplication benchmark
                start_time = time.time()
                a = tf.random.normal([2000, 2000])
                b = tf.random.normal([2000, 2000])
                c = tf.matmul(a, b)
                gpu_time = time.time() - start_time
                
                print(f"✅ GPU computation successful!")
                print(f"   Matrix size: 2000x2000")
                print(f"   GPU time: {gpu_time:.4f}s")
                print(f"   Result shape: {c.shape}")
                
            # CPU comparison
            with tf.device('/CPU:0'):
                start_time = time.time()
                a_cpu = tf.random.normal([1000, 1000])  # Smaller for CPU
                b_cpu = tf.random.normal([1000, 1000])
                c_cpu = tf.matmul(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                
                print(f"   CPU time (1000x1000): {cpu_time:.4f}s")
                speedup = (cpu_time * 4) / gpu_time  # Adjust for matrix size difference
                print(f"   Estimated GPU speedup: {speedup:.2f}x")
                
            return True
        else:
            print("❌ No GPU devices detected")
            return False
            
    except Exception as e:
        print(f"❌ TensorFlow GPU test failed: {e}")
        return False

def test_multimodal_libraries():
    """Test multimodal AI libraries."""
    print("\n🔬 Testing Multimodal AI Libraries")
    print("=" * 40)
    
    tests = []
    
    # Computer Vision
    try:
        import cv2
        import mediapipe as mp
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ MediaPipe: {mp.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Computer Vision libraries: {e}")
        tests.append(False)
    
    # Audio Processing
    try:
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
        
        # Test audio processing
        sr = 22050
        duration = 1  # 1 second
        y = np.sin(2 * np.pi * 440 * np.linspace(0, duration, sr))  # 440 Hz tone
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"   MFCC extraction test: {mfccs.shape}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Audio processing: {e}")
        tests.append(False)
    
    # NLP
    try:
        from transformers import pipeline
        print(f"✅ Transformers: Available")
        
        # Test sentiment analysis pipeline
        classifier = pipeline("sentiment-analysis")
        result = classifier("This is a great setup for multimodal AI!")
        print(f"   Sentiment test: {result[0]['label']}")
        tests.append(True)
    except Exception as e:
        print(f"❌ NLP libraries: {e}")
        tests.append(False)
    
    # Web Frameworks
    try:
        import streamlit
        import fastapi
        print(f"✅ Streamlit: {streamlit.__version__}")
        print(f"✅ FastAPI: {fastapi.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Web frameworks: {e}")
        tests.append(False)
    
    return all(tests)

def main():
    """Run all tests."""
    print("🚀 WSL2 Multimodal AI Setup - Comprehensive Test")
    print("=" * 60)
    
    # System info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Run tests
    gpu_test = test_tensorflow_gpu()
    lib_test = test_multimodal_libraries()
    
    # Results
    print(f"\n📊 Test Results:")
    print(f"   GPU Performance: {'✅ PASS' if gpu_test else '❌ FAIL'}")
    print(f"   Multimodal Libraries: {'✅ PASS' if lib_test else '❌ FAIL'}")
    
    if gpu_test and lib_test:
        print("\n🎉 All tests passed! WSL2 setup is perfect!")
        print("🚀 Ready to build your multimodal mental health AI!")
        print("\n💡 Pro tip: Start with 'jupyter lab --ip=0.0.0.0' for development")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")
    
    return gpu_test and lib_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
