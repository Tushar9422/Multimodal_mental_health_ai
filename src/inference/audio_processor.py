#!/usr/bin/env python3
"""
Audio processing for real-time inference.
Robust multi-backend audio loading (soundfile + librosa + scipy).
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import os
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config

config = get_config()


class AudioProcessor:
    """Process audio files for emotion recognition."""
    
    def __init__(self, sample_rate=22050, n_mfcc=40, max_length=100):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sampling rate
            n_mfcc: Number of MFCC coefficients
            max_length: Maximum time steps
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        
        print(f"🎵 Audio Processor initialized")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   MFCCs: {n_mfcc}")
        print(f"   Max length: {max_length} frames")
    
    def load_audio_with_scipy(self, file_path):
        """
        Load audio using scipy (fallback for difficult WAV files).
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal and sample rate
        """
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(file_path)
            
            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            
            return audio, sr
        except Exception as e:
            raise ValueError(f"Scipy loading failed: {e}")
    
    def load_audio(self, audio_input):
        """
        Load audio file with multi-backend fallback.
        Tries: soundfile → scipy → librosa (in order).
        
        Args:
            audio_input: Path to audio file or Streamlit UploadedFile
            
        Returns:
            Audio signal and sample rate
        """
        temp_path = None
        
        try:
            # Handle file path directly
            if isinstance(audio_input, (str, Path)):
                file_path = str(audio_input)
                
                # Try soundfile first
                try:
                    audio, sr = sf.read(file_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                    return audio, sr
                except:
                    pass
                
                # Try scipy
                try:
                    return self.load_audio_with_scipy(file_path)
                except:
                    pass
                
                # Try librosa as last resort
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, sr = librosa.load(file_path, sr=self.sample_rate)
                    return audio, sr
            
            # Handle Streamlit upload - save to temp file
            file_name = getattr(audio_input, 'name', 'audio.wav')
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if not file_ext:
                file_ext = '.wav'
            
            # Get file data
            file_data = audio_input.getvalue()
            
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
            
            try:
                # Write data
                with os.fdopen(temp_fd, 'wb') as tmp_file:
                    tmp_file.write(file_data)
                
                # Try multiple loading methods
                # Method 1: soundfile
                try:
                    audio, sr = sf.read(temp_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                    return audio, sr
                except Exception as sf_error:
                    print(f"   ⚠️  Soundfile failed: {sf_error}")
                
                # Method 2: scipy
                try:
                    audio, sr = self.load_audio_with_scipy(temp_path)
                    return audio, sr
                except Exception as scipy_error:
                    print(f"   ⚠️  Scipy failed: {scipy_error}")
                
                # Method 3: librosa with audioread
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio, sr = librosa.load(temp_path, sr=self.sample_rate)
                        return audio, sr
                except Exception as librosa_error:
                    print(f"   ⚠️  Librosa failed: {librosa_error}")
                    raise ValueError(
                        f"All audio loading methods failed. "
                        f"File: {file_name}, "
                        f"Format may not be supported."
                    )
            
            finally:
                # Cleanup
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    def extract_mfccs(self, audio, sr=None):
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate (optional)
            
        Returns:
            MFCC features (n_mfcc, max_length)
        """
        if sr is None:
            sr = self.sample_rate
        
        # Ensure audio is not empty
        if len(audio) == 0:
            raise ValueError("Audio signal is empty")
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc
        )
        
        # Pad or truncate to fixed length
        if mfccs.shape[1] < self.max_length:
            # Pad
            pad_width = self.max_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfccs = mfccs[:, :self.max_length]
        
        return mfccs
    
    def process(self, audio_input):
        """
        Complete processing pipeline: audio → MFCCs.
        
        Args:
            audio_input: Audio file path or Streamlit UploadedFile
            
        Returns:
            MFCC features ready for model input (1, max_length, n_mfcc)
        """
        # Load audio
        audio, sr = self.load_audio(audio_input)
        
        # Extract MFCCs
        mfccs = self.extract_mfccs(audio, sr)
        
        # Transpose to (time, features) and add batch dimension
        mfccs = mfccs.T  # (max_length, n_mfcc)
        mfccs = np.expand_dims(mfccs, axis=0)  # (1, max_length, n_mfcc)
        
        return mfccs
    
    def get_audio_info(self, audio_input):
        """
        Get audio file information.
        
        Returns:
            Dictionary with audio metadata
        """
        try:
            audio, sr = self.load_audio(audio_input)
            
            duration = len(audio) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'channels': 1  # Mono after conversion
            }
        except Exception as e:
            return {'error': str(e), 'duration': 0, 'sample_rate': 0, 'samples': 0}


if __name__ == "__main__":
    # Test audio processor
    print("🧪 Testing Audio Processor")
    print("="*60)
    
    processor = AudioProcessor()
    
    # Test with a sample file if available
    sample_dir = config.DATA_DIR / "audio" / "Actor_01"
    
    if sample_dir.exists():
        sample_files = list(sample_dir.glob("*.wav"))
        if sample_files:
            test_file = sample_files[0]
            print(f"\n📂 Testing with: {test_file.name}")
            
            # Get info
            info = processor.get_audio_info(test_file)
            if 'error' not in info:
                print(f"\n📊 Audio Info:")
                print(f"   Duration: {info['duration']:.2f} seconds")
                print(f"   Sample rate: {info['sample_rate']} Hz")
                
                # Process
                mfccs = processor.process(test_file)
                print(f"\n✅ MFCCs extracted:")
                print(f"   Shape: {mfccs.shape}")
                print(f"   Expected: (1, 100, 40)")
            else:
                print(f"\n❌ Error: {info['error']}")
    else:
        print("\n⚠️  No sample audio files found")
