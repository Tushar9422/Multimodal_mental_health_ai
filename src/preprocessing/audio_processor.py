"""
Audio preprocessing and feature extraction for emotion recognition.
Converts audio files into numerical features for model training.
"""

import numpy as np
import librosa
import json
from pathlib import Path
from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import get_project_logger, ensure_dir

logger = get_project_logger("audio_processor")
config = get_config()


class AudioFeatureExtractor:
    """
    Extract features from audio files for emotion recognition.
    
    Features extracted:
    - MFCC: Mel-Frequency Cepstral Coefficients (voice timbre)
    - Mel Spectrogram: Frequency content over time
    - Chroma: Pitch class distribution
    - Spectral features: Centroid, bandwidth, rolloff
    """
    
    def __init__(self, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        print(f"🎵 Audio Feature Extractor Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Duration: {duration} seconds")
        print(f"   Target Length: {self.target_length} samples")
    
    def load_audio(self, file_path):
        """
        Load audio file and ensure consistent length.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio array of fixed length
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Ensure consistent length
            if len(y_trimmed) > self.target_length:
                y_trimmed = y_trimmed[:self.target_length]
            else:
                y_trimmed = np.pad(y_trimmed, (0, self.target_length - len(y_trimmed)))
            
            return y_trimmed
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features (voice characteristics).
        
        MFCCs capture the shape of the vocal tract - good for emotion!
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=40,  # 40 coefficients
            n_fft=2048,
            hop_length=512
        )
        return mfcc
    
    def extract_mel_spectrogram(self, audio):
        """
        Extract Mel-Spectrogram (frequency content over time).
        
        Shows which frequencies are present at each time - like a picture of sound.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,  # 128 mel bands
            n_fft=2048,
            hop_length=512
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma(self, audio):
        """
        Extract Chroma features (pitch information).
        
        Represents 12 pitch classes - useful for emotional prosody.
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512
        )
        return chroma
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral features (frequency characteristics).
        
        - Centroid: "brightness" of sound
        - Bandwidth: frequency spread
        - Rolloff: frequency below which 85% of energy is contained
        """
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )[0]
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        )[0]
        
        return {
            'centroid': spectral_centroids,
            'bandwidth': spectral_bandwidth,
            'rolloff': spectral_rolloff
        }
    
    def extract_all_features(self, audio):
        """
        Extract all features from audio.
        
        Returns:
            Dictionary with all feature arrays
        """
        features = {}
        
        # MFCC (40 x time_steps)
        features['mfcc'] = self.extract_mfcc(audio)
        
        # Mel Spectrogram (128 x time_steps)
        features['mel_spec'] = self.extract_mel_spectrogram(audio)
        
        # Chroma (12 x time_steps)
        features['chroma'] = self.extract_chroma(audio)
        
        # Spectral features (1 x time_steps each)
        spectral = self.extract_spectral_features(audio)
        features['spectral_centroid'] = spectral['centroid']
        features['spectral_bandwidth'] = spectral['bandwidth']
        features['spectral_rolloff'] = spectral['rolloff']
        
        return features
    
    def create_combined_feature_matrix(self, features):
        """
        Combine all features into a single 2D matrix.
        
        This creates an "image" of the audio that CNN can process.
        
        Shape: (183 features, time_steps)
        - 40 MFCC
        - 128 Mel Spectrogram
        - 12 Chroma
        - 3 Spectral (centroid, bandwidth, rolloff)
        """
        feature_list = []
        
        # Stack all features vertically
        feature_list.append(features['mfcc'])  # 40 rows
        feature_list.append(features['mel_spec'])  # 128 rows
        feature_list.append(features['chroma'])  # 12 rows
        feature_list.append(features['spectral_centroid'].reshape(1, -1))  # 1 row
        feature_list.append(features['spectral_bandwidth'].reshape(1, -1))  # 1 row
        feature_list.append(features['spectral_rolloff'].reshape(1, -1))  # 1 row
        
        # Combine into single matrix
        combined = np.vstack(feature_list)
        
        return combined


def prepare_audio_dataset():
    """
    Prepare complete audio dataset for training.
    
    Loads metadata, extracts features from all audio files,
    and saves as numpy arrays for fast training.
    """
    print("🎵 Preparing Audio Dataset for Training")
    print("="*60)
    
    # Load metadata
    metadata_file = config.DATA_DIR / "metadata" / "audio_metadata.json"
    
    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata['samples']
    print(f"📊 Total samples: {len(samples)}")
    
    # Emotion mapping
    emotions = config.AUDIO_EMOTIONS
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    print(f"📋 Emotions: {emotions}")
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor(
        sample_rate=config.AUDIO_SAMPLE_RATE,
        duration=config.AUDIO_DURATION
    )
    
    # Prepare storage
    X_list = []  # Features
    y_list = []  # Labels
    metadata_list = []  # Sample info
    
    print("\n🔄 Extracting features from audio files...")
    
    for sample in tqdm(samples, desc="Processing audio"):
        try:
            # Get file path and emotion
            audio_path = Path(sample['processed_path'])
            emotion = sample['emotion']
            
            if not audio_path.exists():
                logger.warning(f"File not found: {audio_path}")
                continue
            
            # Load audio
            audio = extractor.load_audio(audio_path)
            if audio is None:
                continue
            
            # Extract features
            features = extractor.extract_all_features(audio)
            
            # Create combined feature matrix
            feature_matrix = extractor.create_combined_feature_matrix(features)
            
            # Get emotion label
            emotion_idx = emotion_to_idx.get(emotion, -1)
            if emotion_idx == -1:
                logger.warning(f"Unknown emotion: {emotion}")
                continue
            
            # Store
            X_list.append(feature_matrix)
            y_list.append(emotion_idx)
            metadata_list.append({
                'file_id': sample['file_id'],
                'emotion': emotion,
                'actor': sample.get('actor', 'unknown'),
                'gender': sample.get('gender', 'unknown')
            })
            
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            continue
    
    # Convert to numpy arrays
    print("\n📦 Converting to numpy arrays...")
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Feature extraction complete!")
    print(f"   X shape: {X.shape} (samples, features, time_steps)")
    print(f"   y shape: {y.shape} (samples,)")
    print(f"   Feature dimensions: {X.shape[1]} x {X.shape[2]}")
    
    # Save processed data
    output_dir = ensure_dir(config.DATA_DIR / "processed_features")
    
    print("\n💾 Saving processed data...")
    np.save(output_dir / "X_audio.npy", X)
    np.save(output_dir / "y_audio.npy", y)
    
    with open(output_dir / "audio_metadata.pkl", 'wb') as f:
        pickle.dump(metadata_list, f)
    
    with open(output_dir / "emotion_mapping.json", 'w') as f:
        json.dump({
            'emotions': emotions,
            'emotion_to_idx': emotion_to_idx,
            'idx_to_emotion': {idx: emotion for emotion, idx in emotion_to_idx.items()}
        }, f, indent=2)
    
    print(f"✅ Data saved to: {output_dir}")
    print(f"   - X_audio.npy: Features")
    print(f"   - y_audio.npy: Labels")
    print(f"   - audio_metadata.pkl: Sample metadata")
    print(f"   - emotion_mapping.json: Label mapping")
    
    return True


if __name__ == "__main__":
    success = prepare_audio_dataset()
    
    if success:
        print("\n Audio dataset preparation completed!")
        print("✅ Ready for model training!")
    else:
        print("\n❌ Audio dataset preparation failed")
        sys.exit(1)
