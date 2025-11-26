#!/usr/bin/env python3
"""
Image preprocessing and data loading for facial expression recognition.
Simplified version - Keras handles grayscale → RGB conversion automatically.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir

logger = get_project_logger("image_processor")
config = get_config()


class FER2013DataLoader:
    """Load and preprocess FER-2013 facial expression images."""
    
    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.emotion_labels = config.FACIAL_EMOTIONS
        self.raw_images_dir = config.DATA_DIR / "images" / "raw" / "FER2013"
        
        print(f"👤 FER-2013 Data Loader Initialized")
        print(f"   Image size: {image_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Emotions: {self.emotion_labels}")
    
    def verify_dataset(self):
        """Verify FER-2013 dataset exists."""
        print("\n🔍 Verifying FER-2013 dataset...")
        
        if not self.raw_images_dir.exists():
            print(f"❌ Dataset not found: {self.raw_images_dir}")
            return False
        
        train_dir = self.raw_images_dir / "train"
        test_dir = self.raw_images_dir / "test"
        
        if not train_dir.exists() or not test_dir.exists():
            print(f"❌ train/ or test/ directory missing")
            return False
        
        print(f"✅ Dataset found: {self.raw_images_dir}")
        
        # Count images
        train_count = sum(1 for _ in train_dir.glob("**/*.jpg"))
        test_count = sum(1 for _ in test_dir.glob("**/*.jpg"))
        
        print(f"✅ Train images: {train_count}")
        print(f"✅ Test images: {test_count}")
        
        return True
    
    def create_data_generators(self, validation_split=0.1):
        """
        Create data generators with augmentation.
        Uses color_mode='rgb' which auto-converts grayscale to RGB.
        """
        print("\n🔄 Creating data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            validation_split=validation_split
        )
        
        # Validation/test data (no augmentation)
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_dir = str(self.raw_images_dir / "train")
        test_dir = str(self.raw_images_dir / "test")
        
        # Create generators with color_mode='rgb'
        # Keras automatically converts grayscale → RGB
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            color_mode='rgb'  # Auto-converts grayscale to RGB
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            color_mode='rgb'  # Auto-converts grayscale to RGB
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            color_mode='rgb'  # Auto-converts grayscale to RGB
        )
        
        print(f"\n✅ Data generators created:")
        print(f"   Training samples: {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Test samples: {test_generator.samples}")
        print(f"   Classes: {list(train_generator.class_indices.keys())}")
        print(f"   🔧 Color mode: RGB (auto-converted from grayscale)")
        
        # Save class indices
        class_indices = train_generator.class_indices
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        metadata = {
            'class_indices': class_indices,
            'idx_to_class': idx_to_class,
            'num_classes': len(class_indices),
            'image_size': self.image_size,
            'train_samples': train_generator.samples,
            'val_samples': val_generator.samples,
            'test_samples': test_generator.samples
        }
        
        # Save metadata
        metadata_dir = ensure_dir(config.DATA_DIR / "processed_features")
        with open(metadata_dir / "facial_class_mapping.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Class mapping saved")
        
        return train_generator, val_generator, test_generator


def prepare_facial_dataset():
    """Prepare facial dataset for training."""
    print("👤 Preparing Facial Expression Dataset")
    print("="*60)
    
    loader = FER2013DataLoader(
        image_size=config.IMAGE_SIZE,
        batch_size=32
    )
    
    if not loader.verify_dataset():
        print("\n❌ Dataset verification failed")
        return False
    
    train_gen, val_gen, test_gen = loader.create_data_generators(validation_split=0.1)
    
    print("\n✅ Facial dataset preparation complete!")
    return True


if __name__ == "__main__":
    success = prepare_facial_dataset()
    
    if success:
        print("\n✅ Facial dataset ready for model training!")
    else:
        print("\n❌ Facial dataset preparation failed")
        sys.exit(1)
