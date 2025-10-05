"""
WSL2 Ubuntu TensorFlow configuration for Mental Health AI system.
Optimized for GPU acceleration and professional development.
"""

import os
import tensorflow as tf
from pathlib import Path
from typing import Dict, List

class Config:
    """Main configuration class optimized for WSL2 Ubuntu."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard_logs"
    
    # Data paths
    AUDIO_DATA_DIR = DATA_DIR / "audio"
    VIDEO_DATA_DIR = DATA_DIR / "video"
    TEXT_DATA_DIR = DATA_DIR / "text"
    PHYSIOLOGICAL_DATA_DIR = DATA_DIR / "physiological"
    
    # Model paths
    AUDIO_MODEL_DIR = MODELS_DIR / "audio_emotion"
    FACIAL_MODEL_DIR = MODELS_DIR / "facial_emotion"
    TEXT_MODEL_DIR = MODELS_DIR / "text_sentiment"
    FUSION_MODEL_DIR = MODELS_DIR / "fusion_model"
    
    # TensorFlow Configuration for WSL2
    USE_GPU = True
    GPU_MEMORY_GROWTH = True
    MIXED_PRECISION = True
    XLA_COMPILATION = True
    
    # TensorFlow settings
    TF_CPP_MIN_LOG_LEVEL = "1"
    ENABLE_TENSORBOARD = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_FREQUENCY = 5
    
    # Audio processing settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 3.0
    AUDIO_N_MFCC = 13
    AUDIO_N_CHROMA = 12
    AUDIO_N_MEL = 128
    AUDIO_HOP_LENGTH = 512
    AUDIO_N_FFT = 2048
    
    # Image processing settings
    IMAGE_SIZE = (224, 224)
    IMAGE_CHANNELS = 3
    FACE_DETECTION_CONFIDENCE = 0.5
    
    # Text processing settings
    MAX_TEXT_LENGTH = 512
    TEXT_MODEL_NAME = "distilbert-base-uncased"
    VOCAB_SIZE = 30000
    EMBEDDING_DIM = 128
    
    # Model training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.2
    
    # GPU-optimized batch sizes
    GPU_BATCH_SIZES = {
        'audio_model': 64,
        'facial_model': 32,
        'text_model': 16,
        'fusion_model': 24
    }
    
    # Model architectures
    AUDIO_MODEL_ARCHITECTURE = "cnn_lstm"
    FACIAL_MODEL_ARCHITECTURE = "efficientnet_b0"
    TEXT_MODEL_ARCHITECTURE = "distilbert"
    FUSION_ARCHITECTURE = "attention_fusion"
    
    # Emotion categories
    AUDIO_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    FACIAL_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    TEXT_SENTIMENTS = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    
    # Fusion weights
    FUSION_WEIGHTS = {'audio': 0.3, 'facial': 0.3, 'text': 0.4}
    
    # Real-time processing
    PROCESSING_BUFFER_SIZE = 1024
    FPS = 30
    AUDIO_CHUNK_SIZE = 4096
    
    # Mental health thresholds
    CRISIS_THRESHOLD = 0.8
    INTERVENTION_CATEGORIES = [
        'breathing_exercises', 'mindfulness', 'professional_help',
        'emergency_resources', 'positive_affirmations'
    ]
    
    @classmethod
    def setup_tensorflow_gpu(cls):
        """Configure TensorFlow for WSL2 Ubuntu GPU optimization."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(cls.TF_CPP_MIN_LOG_LEVEL)
        
        gpu_info = {
            'gpu_available': False,
            'gpu_devices': [],
            'memory_growth_set': False,
            'mixed_precision_enabled': False,
            'xla_enabled': False
        }
        
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if physical_devices and cls.USE_GPU:
                gpu_info['gpu_available'] = True
                gpu_info['gpu_devices'] = [device.name for device in physical_devices]
                
                # Configure memory growth (essential for WSL2)
                if cls.GPU_MEMORY_GROWTH:
                    try:
                        for device in physical_devices:
                            tf.config.experimental.set_memory_growth(device, True)
                        gpu_info['memory_growth_set'] = True
                        print("✅ GPU memory growth enabled")
                    except Exception as e:
                        print(f"⚠️ Memory growth setting failed: {e}")
                
                # Enable mixed precision
                if cls.MIXED_PRECISION:
                    try:
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        gpu_info['mixed_precision_enabled'] = True
                        print("✅ Mixed precision enabled")
                    except Exception as e:
                        print(f"⚠️ Mixed precision failed: {e}")
                
                # Enable XLA compilation
                if cls.XLA_COMPILATION:
                    try:
                        tf.config.optimizer.set_jit(True)
                        gpu_info['xla_enabled'] = True
                        print("✅ XLA compilation enabled")
                    except Exception as e:
                        print(f"⚠️ XLA compilation failed: {e}")
                
                # Print GPU information
                for i, device in enumerate(physical_devices):
                    try:
                        details = tf.config.experimental.get_device_details(device)
                        name = details.get('device_name', 'Unknown GPU')
                        print(f"✅ GPU {i}: {name}")
                    except:
                        print(f"✅ GPU {i}: {device.name}")
            else:
                print("❌ No GPU devices found")
                
        except Exception as e:
            print(f"❌ TensorFlow GPU setup failed: {e}")
        
        return gpu_info
    
    @classmethod
    def optimize_batch_size(cls, model_type='facial_model', gpu_memory_gb=8):
        """Optimize batch size for WSL2 GPU."""
        if not cls.USE_GPU:
            return cls.BATCH_SIZE
        
        # WSL2 typically has good GPU access, use optimistic sizes
        memory_batch_map = {
            4: {'audio_model': 32, 'facial_model': 16, 'text_model': 8, 'fusion_model': 12},
            6: {'audio_model': 48, 'facial_model': 24, 'text_model': 12, 'fusion_model': 16},
            8: {'audio_model': 64, 'facial_model': 32, 'text_model': 16, 'fusion_model': 24},
            12: {'audio_model': 96, 'facial_model': 48, 'text_model': 24, 'fusion_model': 32},
            16: {'audio_model': 128, 'facial_model': 64, 'text_model': 32, 'fusion_model': 40},
        }
        
        closest_memory = min(memory_batch_map.keys(), key=lambda x: abs(x - gpu_memory_gb))
        optimized_size = memory_batch_map[closest_memory].get(model_type, cls.BATCH_SIZE)
        
        print(f"🔧 Optimized batch size for {model_type}: {optimized_size}")
        return optimized_size
    
    @classmethod
    def create_directories(cls):
        """Create all project directories."""
        directories = [
            cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR,
            cls.CHECKPOINTS_DIR, cls.TENSORBOARD_DIR,
            cls.AUDIO_DATA_DIR, cls.VIDEO_DATA_DIR,
            cls.TEXT_DATA_DIR, cls.PHYSIOLOGICAL_DATA_DIR,
            cls.AUDIO_MODEL_DIR, cls.FACIAL_MODEL_DIR,
            cls.TEXT_MODEL_DIR, cls.FUSION_MODEL_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ All project directories created!")

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    EPOCHS = 10
    SAVE_CHECKPOINTS = True

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"
    BATCH_SIZE = 64
    EPOCHS = 100

def get_config():
    """Get configuration based on environment."""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    return ProductionConfig() if env == 'production' else DevelopmentConfig()
