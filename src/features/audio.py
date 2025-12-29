"""
Audio Feature Extraction Module
Extracts emotion, energy, pitch, and other audio signals for viral clip detection
"""

import numpy as np
import subprocess
import json
import wave
import struct
from pathlib import Path


class AudioFeatureExtractor:
    """Extract viral-relevant features from audio"""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract audio track from video using ffmpeg"""
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.wav'))
            
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    def load_audio(self, audio_path):
        """Load audio file as numpy array"""
        with wave.open(audio_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            audio_data = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
            
            return audio_array, framerate
    
    def compute_energy(self, audio, window_size=0.5):
        """
        Compute RMS energy over time windows
        High energy = excitement, engagement
        """
        samples_per_window = int(window_size * self.sample_rate)
        n_windows = len(audio) // samples_per_window
        
        energy_timeline = []
        timestamps = []
        
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = audio[start:end]
            
            # RMS energy
            rms = np.sqrt(np.mean(window ** 2))
            energy_timeline.append(rms)
            timestamps.append(i * window_size)
        
        return np.array(energy_timeline), np.array(timestamps)
    
    def compute_zero_crossing_rate(self, audio, window_size=0.5):
        """
        Zero crossing rate - indicates speech vs music/noise
        High ZCR = fricatives, high-frequency content
        """
        samples_per_window = int(window_size * self.sample_rate)
        n_windows = len(audio) // samples_per_window
        
        zcr_timeline = []
        
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = audio[start:end]
            
            # Count zero crossings
            signs = np.sign(window)
            zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(window))
            zcr_timeline.append(zcr)
        
        return np.array(zcr_timeline)
    
    def detect_silence(self, audio, threshold=0.01, window_size=0.5):
        """
        Detect silent regions
        Important: viral clips should minimize dead air
        """
        samples_per_window = int(window_size * self.sample_rate)
        n_windows = len(audio) // samples_per_window
        
        silence_timeline = []
        
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = audio[start:end]
            
            rms = np.sqrt(np.mean(window ** 2))
            is_silent = rms < threshold
            silence_timeline.append(is_silent)
        
        return np.array(silence_timeline)
    
    def compute_spectral_features(self, audio, window_size=0.5):
        """
        Compute spectral centroid and rolloff
        Centroid = brightness of sound
        Rolloff = frequency below which X% of energy is contained
        """
        samples_per_window = int(window_size * self.sample_rate)
        n_windows = len(audio) // samples_per_window
        
        spectral_centroids = []
        spectral_rolloffs = []
        
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = audio[start:end]
            
            # FFT
            spectrum = np.abs(np.fft.rfft(window))
            freqs = np.fft.rfftfreq(len(window), 1/self.sample_rate)
            
            # Spectral centroid
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            spectral_centroids.append(centroid)
            
            # Spectral rolloff (85% threshold)
            cumsum = np.cumsum(spectrum)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            spectral_rolloffs.append(rolloff)
        
        return np.array(spectral_centroids), np.array(spectral_rolloffs)
    
    def estimate_pitch(self, audio, window_size=0.5):
        """
        Estimate fundamental frequency (F0) using autocorrelation
        High pitch variance = emotional intensity, excitement
        """
        samples_per_window = int(window_size * self.sample_rate)
        n_windows = len(audio) // samples_per_window
        
        pitch_timeline = []
        
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = audio[start:end]
            
            # Autocorrelation
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after zero lag
            # Look in range 50-400 Hz (typical human speech)
            min_lag = int(self.sample_rate / 400)
            max_lag = int(self.sample_rate / 50)
            
            autocorr_subset = autocorr[min_lag:max_lag]
            if len(autocorr_subset) > 0:
                peak_idx = np.argmax(autocorr_subset) + min_lag
                pitch = self.sample_rate / peak_idx if peak_idx > 0 else 0
            else:
                pitch = 0
            
            pitch_timeline.append(pitch)
        
        return np.array(pitch_timeline)
    
    def detect_emotional_peaks(self, energy, pitch, window_size=5):
        """
        Detect emotional peaks based on energy and pitch variance
        Peaks indicate exciting moments worth extracting
        """
        # Normalize features
        energy_norm = (energy - np.mean(energy)) / (np.std(energy) + 1e-10)
        pitch_variance = np.array([np.std(pitch[max(0, i-window_size):i+window_size+1]) 
                                   for i in range(len(pitch))])
        pitch_var_norm = (pitch_variance - np.mean(pitch_variance)) / (np.std(pitch_variance) + 1e-10)
        
        # Combined emotional intensity
        emotion_intensity = 0.6 * energy_norm + 0.4 * pitch_var_norm
        
        # Find peaks (above 1 std)
        threshold = 1.0
        peaks = emotion_intensity > threshold
        
        return emotion_intensity, peaks
    
    def extract_all_features(self, video_path, window_size=0.5):
        """
        Extract all audio features for viral clip detection
        
        Returns:
            dict with timestamps and feature arrays
        """
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        audio, sr = self.load_audio(audio_path)
        
        # Compute features
        energy, timestamps = self.compute_energy(audio, window_size)
        zcr = self.compute_zero_crossing_rate(audio, window_size)
        silence = self.detect_silence(audio, window_size=window_size)
        spectral_centroid, spectral_rolloff = self.compute_spectral_features(audio, window_size)
        pitch = self.estimate_pitch(audio, window_size)
        emotion_intensity, emotion_peaks = self.detect_emotional_peaks(energy, pitch)
        
        features = {
            'timestamps': timestamps.tolist(),
            'energy': energy.tolist(),
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)),
            'zero_crossing_rate': zcr.tolist(),
            'silence': silence.tolist(),
            'silence_ratio': float(np.mean(silence)),
            'spectral_centroid': spectral_centroid.tolist(),
            'spectral_rolloff': spectral_rolloff.tolist(),
            'pitch': pitch.tolist(),
            'pitch_mean': float(np.mean(pitch[pitch > 0])) if np.any(pitch > 0) else 0,
            'pitch_variance': float(np.var(pitch[pitch > 0])) if np.any(pitch > 0) else 0,
            'emotion_intensity': emotion_intensity.tolist(),
            'emotion_peaks': emotion_peaks.tolist(),
            'sample_rate': sr,
            'audio_path': audio_path
        }
        
        return features


if __name__ == '__main__':
    # Test
    extractor = AudioFeatureExtractor()
    # features = extractor.extract_all_features('test_video.mp4')
    # print(f"Extracted {len(features['timestamps'])} time windows")
    # print(f"Silence ratio: {features['silence_ratio']:.2%}")
    # print(f"Mean energy: {features['energy_mean']:.3f}")
    print("Audio feature extractor ready")
