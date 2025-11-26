#!/usr/bin/env python3
"""
Image processing for real-time inference.
Handles image file → preprocessing → prediction.
FIXED: Properly handles Streamlit UploadedFile objects.
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config

config = get_config()


class ImageProcessor:
    """Process images for facial emotion recognition."""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
        print(f"📸 Image Processor initialized")
        print(f"   Target size: {target_size}")
    
    def load_image(self, image_input):
        """
        Load image from file or Streamlit UploadedFile.
        
        Args:
            image_input: Image file path or Streamlit UploadedFile
            
        Returns:
            PIL Image
        """
        try:
            if isinstance(image_input, (str, Path)):
                # File path
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                # Already a PIL Image
                img = image_input
            else:
                # Streamlit UploadedFile or file-like object
                # Get bytes directly using getvalue()
                image_bytes = image_input.getvalue()
                
                # Create PIL Image from bytes
                img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
        
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def preprocess_image(self, img):
        """
        Preprocess image for model input.
        
        Args:
            img: PIL Image
            
        Returns:
            Preprocessed image array
        """
        # Resize
        img = img.resize(self.target_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def process(self, image_input):
        """
        Complete processing pipeline: image → preprocessed.
        
        Args:
            image_input: Image file path or Streamlit UploadedFile
            
        Returns:
            Image array ready for model input (1, 224, 224, 3)
        """
        # Load image
        img = self.load_image(image_input)
        
        # Preprocess
        img_array = self.preprocess_image(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        
        return img_array
    
    def get_image_info(self, image_input):
        """
        Get image metadata.
        
        Returns:
            Dictionary with image info
        """
        try:
            img = self.load_image(image_input)
            
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'width': img.width,
                'height': img.height
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_face(self, image_input):
        """
        Detect face in image (optional enhancement).
        
        Returns:
            Cropped face image or original if no face detected
        """
        try:
            img = self.load_image(image_input)
            img_array = np.array(img)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Crop face with margin
                margin = int(0.2 * min(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img.width, x + w + margin)
                y2 = min(img.height, y + h + margin)
                
                face_img = img.crop((x1, y1, x2, y2))
                return face_img
            else:
                # No face detected, return original
                return img
        
        except Exception as e:
            print(f"⚠️  Face detection failed: {e}")
            return self.load_image(image_input)


if __name__ == "__main__":
    # Test image processor
    print("🧪 Testing Image Processor")
    print("="*60)
    
    processor = ImageProcessor()
    
    # Test with a sample file if available
    sample_dir = config.DATA_DIR / "images" / "raw" / "test" / "happy"
    
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if sample_files:
            test_file = sample_files[0]
            print(f"\n📂 Testing with: {test_file.name}")
            
            # Get info
            info = processor.get_image_info(test_file)
            print(f"\n📊 Image Info:")
            print(f"   Size: {info['size']}")
            print(f"   Mode: {info['mode']}")
            
            # Process
            img_array = processor.process(test_file)
            print(f"\n✅ Image processed:")
            print(f"   Shape: {img_array.shape}")
            print(f"   Expected: (1, 224, 224, 3)")
            print(f"   Value range: [{img_array.min():.2f}, {img_array.max():.2f}]")
    else:
        print("\n⚠️  No sample images found")
        print(f"   Place test images in: {sample_dir}")
