import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("facial_collector")
config = get_config()

class FER2013FacialCollector:
    """Collect and preprocess FER-2013 facial expression data (folder-based)."""
    
    def __init__(self):
        self.config = config
        
        # Use images directory instead of video
        self.raw_images_dir = config.DATA_DIR / "images" / "raw"
        self.processed_images_dir = config.DATA_DIR / "images" / "processed"
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # FER-2013 emotion mapping (folder names to standardized labels)
        self.fer_emotions = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
        ensure_dir(self.raw_images_dir)
        ensure_dir(self.processed_images_dir)
        
        logger.info(f"Raw images directory: {self.raw_images_dir}")
        logger.info(f"Processed images directory: {self.processed_images_dir}")
    
    def verify_fer2013_exists(self):
        """Verify that FER-2013 dataset exists in correct folder structure."""
        fer_path = self.raw_images_dir / "FER2013"
        
        # Check for train and test directories
        train_dir = fer_path / "train"
        test_dir = fer_path / "test"
        
        if not train_dir.exists():
            logger.error(f"Train directory not found: {train_dir}")
            return False
        
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            return False
        
        logger.info(f" Found train directory: {train_dir}")
        logger.info(f" Found test directory: {test_dir}")
        
        # Check for emotion subdirectories in train
        emotion_dirs = list(train_dir.glob("*"))
        logger.info(f" Found {len(emotion_dirs)} emotion categories in train")
        
        # Count total images
        train_images = list(train_dir.glob("**/*.jpg")) + list(train_dir.glob("**/*.png"))
        test_images = list(test_dir.glob("**/*.jpg")) + list(test_dir.glob("**/*.png"))
        
        logger.info(f" Train images: {len(train_images)}")
        logger.info(f" Test images: {len(test_images)}")
        logger.info(f" Total images: {len(train_images) + len(test_images)}")
        
        if len(train_images) == 0 and len(test_images) == 0:
            logger.error("No images found in train or test directories")
            return False
        
        return True
    
    def detect_and_extract_face(self, image):
        """
        Detect and extract face from image using MediaPipe.
        Returns processed face image or None if no face detected.
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
            ) as face_detection:
                
                results = face_detection.process(rgb_image)
                
                if results.detections:
                    # Get first detection (primary face)
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    
                    h, w, _ = image.shape
                    
                    # Convert relative coordinates to pixels
                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)
                    
                    # Add padding (10% on each side)
                    padding_x = int(width * 0.1)
                    padding_y = int(height * 0.1)
                    
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    width = min(w - x, width + 2 * padding_x)
                    height = min(h - y, height + 2 * padding_y)
                    
                    # Crop face region
                    face = image[y:y+height, x:x+width]
                    
                    # Resize to standard size (224x224)
                    face_resized = cv2.resize(face, config.IMAGE_SIZE)
                    
                    return face_resized
                else:
                    # No face detected - just resize original image
                    return cv2.resize(image, config.IMAGE_SIZE)
        
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            # Fallback: just resize
            return cv2.resize(image, config.IMAGE_SIZE)
    
    def extract_facial_landmarks(self, face_image):
        """Extract 468 facial landmarks using MediaPipe Face Mesh."""
        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                
                results = face_mesh.process(rgb_image)
                
                if results.multi_face_landmarks:
                    landmarks = []
                    for face_landmarks in results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    return landmarks
        
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
        
        return None
    
    def process_single_image(self, image_path, emotion_label, split):
        """
        Process a single facial image.
        
        Args:
            image_path: Path to the image file
            emotion_label: Emotion category (folder name)
            split: 'train' or 'test'
        
        Returns:
            Dictionary with processed data or None if processing failed
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None
            
            # Detect and extract face
            face = self.detect_and_extract_face(image)
            
            if face is None:
                logger.warning(f"No face detected in: {image_path}")
                return None
            
            # Extract facial landmarks
            # landmarks = self.extract_facial_landmarks(face)
            
            # Generate output filename
            relative_path = image_path.relative_to(self.raw_images_dir / "FER2013")
            output_filename = f"fer2013_{split}_{emotion_label}_{image_path.stem}.jpg"
            output_path = self.processed_images_dir / output_filename
            
            # Save processed image
            cv2.imwrite(str(output_path), face)
            
            # Create sample metadata
            sample_data = {
                'file_id': f"fer2013_{split}_{image_path.stem}",
                'dataset': 'FER2013',
                'emotion': emotion_label,
                'split': split,
                'original_path': str(image_path),
                'processed_path': str(output_path),
                'image_shape': list(config.IMAGE_SIZE) + [3],
                # 'landmarks': landmarks if landmarks else None
            }
            
            return sample_data
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def process_fer2013_split(self, split_name):
        """
        Process all images in a split (train or test).
        
        Args:
            split_name: 'train' or 'test'
        
        Returns:
            List of processed sample metadata
        """
        logger.info(f"Processing FER-2013 {split_name} split...")
        
        fer_path = self.raw_images_dir / "FER2013"
        split_dir = fer_path / split_name
        
        if not split_dir.exists():
            logger.error(f"Split directory not found: {split_dir}")
            return []
        
        processed_data = []
        failed_count = 0
        
        # Process each emotion category
        for emotion_dir in sorted(split_dir.glob("*")):
            if not emotion_dir.is_dir():
                continue
            
            emotion_label = emotion_dir.name.lower()
            
            # Validate emotion label
            if emotion_label not in self.fer_emotions:
                logger.warning(f"Unknown emotion category: {emotion_label}")
                continue
            
            # Get all images in this emotion category
            image_files = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
            
            logger.info(f"Processing {emotion_label}: {len(image_files)} images")
            
            # Process each image with progress bar
            for image_path in tqdm(image_files, desc=f"{split_name}/{emotion_label}", leave=False):
                sample_data = self.process_single_image(image_path, emotion_label, split_name)
                
                if sample_data:
                    processed_data.append(sample_data)
                else:
                    failed_count += 1
        
        logger.info(f"{split_name} split - Successfully processed: {len(processed_data)}, Failed: {failed_count}")
        
        return processed_data
    
    def process_complete_dataset(self):
        """Process both train and test splits."""
        logger.info("Processing complete FER-2013 dataset...")
        
        all_processed_data = []
        
        with Timer("Processing FER-2013 dataset"):
            # Process train split
            train_data = self.process_fer2013_split("train")
            all_processed_data.extend(train_data)
            
            # Process test split
            test_data = self.process_fer2013_split("test")
            all_processed_data.extend(test_data)
        
        logger.info(f"Total processed images: {len(all_processed_data)}")
        
        return all_processed_data
    
    def create_facial_metadata(self, processed_data):
        """Create comprehensive facial expression metadata."""
        metadata_file = config.DATA_DIR / "metadata" / "facial_metadata.json"
        ensure_dir(metadata_file.parent)
        
        # Create summary statistics
        summary = {
            'total_samples': len(processed_data),
            'dataset': 'FER2013',
            'emotions': {},
            'splits': {}
        }
        
        # Count by emotion and split
        for sample in processed_data:
            emotion = sample['emotion']
            split = sample['split']
            
            summary['emotions'][emotion] = summary['emotions'].get(emotion, 0) + 1
            summary['splits'][split] = summary['splits'].get(split, 0) + 1
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'summary': summary,
                'samples': processed_data
            }, f, indent=2)
        
        logger.info(f"Facial metadata saved to {metadata_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("FER-2013 Dataset Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  Emotion distribution:")
        for emotion, count in sorted(summary['emotions'].items()):
            logger.info(f"    {emotion}: {count}")
        logger.info(f"  Split distribution:")
        for split, count in summary['splits'].items():
            logger.info(f"    {split}: {count}")
        logger.info("="*50)
        
        return summary
    
    def collect_and_process(self):
        """Main method to collect and process FER-2013 facial data."""
        logger.info("Starting FER-2013 facial expression data collection...")
        
        # Step 1: Verify dataset exists
        if not self.verify_fer2013_exists():
            return False
        
        # Step 2: Process complete dataset
        processed_data = self.process_complete_dataset()
        
        if not processed_data:
            logger.error("No facial data was processed successfully")
            return False
        
        # Step 3: Create metadata
        summary = self.create_facial_metadata(processed_data)
        
        logger.info("Facial expression processing complete!")
        return True

def main():
    """Run FER-2013 facial data collection."""
    print(" FER-2013 Facial Expression Data Collection")
    print("=" * 60)
    
    collector = FER2013FacialCollector()
    success = collector.collect_and_process()
    
    if success:
        print("\n FER-2013 facial data collection completed successfully!")
    else:
        print("\n FER-2013 facial data collection failed")
        return False
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
