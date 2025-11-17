import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir, Timer

logger = get_project_logger("audio_collector")
config = get_config()

class RAVDESSAudioCollector:
    
    def __init__(self):
        self.config = config
        self.raw_audio_dir = config.AUDIO_DATA_DIR / "raw"
        self.processed_audio_dir = config.AUDIO_DATA_DIR / "processed"
        
        # RAVDESS emotion mapping (from filename codes)
        self.ravdess_emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # RAVDESS intensity mapping
        self.intensity_map = {
            '01': 'normal',
            '02': 'strong'
        }
        
        ensure_dir(self.raw_audio_dir)
        ensure_dir(self.processed_audio_dir)
        
        logger.info(f"Raw audio directory: {self.raw_audio_dir}")
        logger.info(f"Processed audio directory: {self.processed_audio_dir}")
    
    def verify_ravdess_exists(self):
        """Verify that RAVDESS dataset exists in the correct location."""
        ravdess_path = self.raw_audio_dir / "RAVDESS"
        
        
        # Check for actor directories
        actor_dirs = list(ravdess_path.glob("Actor_*"))
        
        if len(actor_dirs) == 0:
            logger.error("No Actor directories found in RAVDESS folder")
            logger.info("Expected structure: RAVDESS/Actor_01/, Actor_02/, etc.")
            return False
        
        logger.info(f" Found {len(actor_dirs)} actor directories")
        
        # Count audio files
        audio_files = list(ravdess_path.glob("**/*.wav"))
        logger.info(f" Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            logger.error("No .wav files found in RAVDESS directories")
            return False
        
        return True
    
    def parse_ravdess_filename(self, filepath):
        """
        Parse RAVDESS filename to extract metadata.
        Returns dict with metadata or None if invalid.
        """
        try:
            filename = filepath.stem  # Get filename without extension
            parts = filename.split('-')
            
            if len(parts) != 7:
                logger.warning(f"Invalid filename format: {filename}")
                return None
            
            modality_code = parts[0]
            vocal_channel = parts[1]
            emotion_code = parts[2]
            intensity_code = parts[3]
            statement = parts[4]
            repetition = parts[5]
            actor = parts[6]
            
            # Validate modality
            if modality_code != '03':
                logger.warning(f"Unexpected modality {modality_code} in {filename}")
                return None
            
            metadata = {
                'file_id': filename,
                'modality': 'audio',
                'vocal_channel': 'speech' if vocal_channel == '01' else 'song',
                'emotion_code': emotion_code,
                'emotion': self.ravdess_emotions.get(emotion_code, 'unknown'),
                'intensity_code': intensity_code,
                'intensity': self.intensity_map.get(intensity_code, 'unknown'),
                'statement': f"statement_{statement}",
                'repetition': int(repetition),
                'actor': int(actor),
                'gender': 'female' if int(actor) % 2 == 0 else 'male',  # Even = female, Odd = male
                'original_path': str(filepath)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing filename {filepath}: {e}")
            return None
    
    def extract_audio_features(self, file_path):
        """
        Extract comprehensive audio features from a single file.
        Returns: (features_dict, processed_audio_array) or (None, None) on error
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=config.AUDIO_SAMPLE_RATE)
            
            # Trim silence from beginning and end
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Ensure consistent duration (pad or trim)
            target_length = int(config.AUDIO_DURATION * sr)
            if len(y_trimmed) > target_length:
                # Trim to target length
                y_trimmed = y_trimmed[:target_length]
            else:
                # Pad with zeros
                y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
            
            # Initialize features dictionary
            features = {}
            
            # 1. MFCC (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(
                y=y_trimmed, 
                sr=sr, 
                n_mfcc=config.AUDIO_N_MFCC,
                hop_length=config.AUDIO_HOP_LENGTH,
                n_fft=config.AUDIO_N_FFT
            )
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            features['mfcc_max'] = np.max(mfcc, axis=1).tolist()
            features['mfcc_min'] = np.min(mfcc, axis=1).tolist()
            
            # 2. Chroma Features
            chroma = librosa.feature.chroma_stft(
                y=y_trimmed, 
                sr=sr,
                hop_length=config.AUDIO_HOP_LENGTH,
                n_fft=config.AUDIO_N_FFT
            )
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            features['chroma_std'] = np.std(chroma, axis=1).tolist()
            
            # 3. Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y_trimmed, 
                sr=sr,
                n_mels=config.AUDIO_N_MEL,
                hop_length=config.AUDIO_HOP_LENGTH,
                n_fft=config.AUDIO_N_FFT
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_mean'] = np.mean(mel_spec_db).item()
            features['mel_std'] = np.std(mel_spec_db).item()
            
            # 4. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids).item()
            features['spectral_centroid_std'] = np.std(spectral_centroids).item()
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth).item()
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff).item()
            
            # 5. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
            features['zcr_mean'] = np.mean(zcr).item()
            features['zcr_std'] = np.std(zcr).item()
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=y_trimmed)[0]
            features['rms_mean'] = np.mean(rms).item()
            features['rms_std'] = np.std(rms).item()
            
            # 7. Tempo
            tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
            features['tempo'] = float(tempo)
            
            return features, y_trimmed
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            return None, None
    
    def process_ravdess_dataset(self):
        """Process complete RAVDESS dataset."""
        logger.info("Processing RAVDESS dataset...")
        
        ravdess_path = self.raw_audio_dir / "RAVDESS"
        
        # Get all audio files
        audio_files = sorted(list(ravdess_path.glob("**/*.wav")))
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        processed_data = []
        failed_count = 0
        
        with Timer(f"Processing {len(audio_files)} RAVDESS audio files"):
            for audio_file in tqdm(audio_files, desc="Processing RAVDESS"):
                # Parse filename for metadata
                metadata = self.parse_ravdess_filename(audio_file)
                if not metadata:
                    failed_count += 1
                    continue
                
                # Extract audio features
                features, audio_data = self.extract_audio_features(audio_file)
                if features is None or audio_data is None:
                    failed_count += 1
                    continue
                
                # Save processed audio file
                output_filename = f"ravdess_{audio_file.stem}_processed.wav"
                output_path = self.processed_audio_dir / output_filename
                
                try:
                    sf.write(output_path, audio_data, config.AUDIO_SAMPLE_RATE)
                except Exception as e:
                    logger.error(f"Error saving processed audio: {e}")
                    failed_count += 1
                    continue
                
                # Combine metadata and features
                sample_data = {
                    'dataset': 'RAVDESS',
                    'processed_path': str(output_path),
                    **metadata,
                    **features
                }
                
                processed_data.append(sample_data)
        
        logger.info(f"Successfully processed: {len(processed_data)} files")
        logger.info(f"Failed to process: {failed_count} files")
        
        return processed_data
    
    def create_audio_metadata(self, processed_data):
        """Create comprehensive metadata file."""
        metadata_file = config.DATA_DIR / "metadata" / "audio_metadata.json"
        ensure_dir(metadata_file.parent)
        
        # Create summary statistics
        summary = {
            'total_samples': len(processed_data),
            'dataset': 'RAVDESS',
            'emotions': {},
            'intensity': {},
            'gender': {},
            'actors': set()
        }
        
        # Count by emotion, intensity, gender, and actor
        for sample in processed_data:
            emotion = sample['emotion']
            intensity = sample['intensity']
            gender = sample['gender']
            actor = sample['actor']
            
            summary['emotions'][emotion] = summary['emotions'].get(emotion, 0) + 1
            summary['intensity'][intensity] = summary['intensity'].get(intensity, 0) + 1
            summary['gender'][gender] = summary['gender'].get(gender, 0) + 1
            summary['actors'].add(actor)
        
        # Convert set to list for JSON serialization
        summary['actors'] = sorted(list(summary['actors']))
        summary['num_actors'] = len(summary['actors'])
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump({
                'summary': summary,
                'samples': processed_data
            }, f, indent=2)
        
        logger.info(f"Audio metadata saved to {metadata_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("RAVDESS Dataset Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  Actors: {summary['num_actors']}")
        logger.info(f"  Emotions distribution:")
        for emotion, count in summary['emotions'].items():
            logger.info(f"    {emotion}: {count}")
        logger.info(f"  Gender distribution:")
        for gender, count in summary['gender'].items():
            logger.info(f"    {gender}: {count}")
        logger.info("="*50)
        
        return summary
    
    def collect_and_process(self):
        """Main method to collect and process RAVDESS audio data."""
        logger.info("Starting RAVDESS audio data collection...")
        
        # Step 1: Verify dataset exists
        if not self.verify_ravdess_exists():
            return False
        
        # Step 2: Process dataset
        processed_data = self.process_ravdess_dataset()
        
        if not processed_data:
            logger.error("No audio data was processed successfully")
            return False
        
        # Step 3: Create metadata
        summary = self.create_audio_metadata(processed_data)
        
        logger.info("Audio processing complete!")
        return True

def main():
    """Run RAVDESS audio data collection."""
    print("🎵 RAVDESS Audio Data Collection")
    print("=" * 60)
    
    collector = RAVDESSAudioCollector()
    success = collector.collect_and_process()
    
    if success:
        print("\n✅ RAVDESS audio data collection completed successfully!")
        print("\nNext steps:")
        print("1. Check data/audio/processed/ for processed audio files")
        print("2. Review data/metadata/audio_metadata.json for statistics")
        print("3. Proceed with facial and text data collection")
    else:
        print("\n❌ RAVDESS audio data collection failed")
        print("Please check the error messages above")
        return False
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
