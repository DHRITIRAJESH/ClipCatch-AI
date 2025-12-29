"""
Visual Feature Extraction Module
Extracts scene changes, motion, and visual interest signals
"""

import numpy as np
import subprocess
import json
import cv2
from pathlib import Path


class VisualFeatureExtractor:
    """Extract visual features for viral clip detection"""
    
    def __init__(self, sample_rate=1.0):
        """
        Args:
            sample_rate: Frames per second to analyze (1.0 = every second)
        """
        self.sample_rate = sample_rate
        
    def get_video_info(self, video_path):
        """Get video metadata using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in info['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        fps = eval(video_stream['r_frame_rate'])  # e.g., "30/1"
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        duration = float(info['format']['duration'])
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def detect_scene_changes(self, video_path, threshold=30.0):
        """
        Detect scene changes (cuts, transitions)
        Important for clip boundary detection
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames according to sample_rate
        frame_skip = int(fps / self.sample_rate) if self.sample_rate < fps else 1
        
        scene_changes = []
        timestamps = []
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Compute frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)
                    
                    is_scene_change = mean_diff > threshold
                    scene_changes.append(is_scene_change)
                    timestamps.append(frame_idx / fps)
                
                prev_frame = gray
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(scene_changes), np.array(timestamps)
    
    def compute_motion_intensity(self, video_path):
        """
        Compute motion intensity using optical flow
        High motion = action, engagement
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps / self.sample_rate) if self.sample_rate < fps else 1
        
        motion_intensities = []
        timestamps = []
        prev_gray = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Compute optical flow (Farneback method)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray,
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0
                    )
                    
                    # Magnitude of flow
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    motion_intensity = np.mean(magnitude)
                    
                    motion_intensities.append(motion_intensity)
                    timestamps.append(frame_idx / fps)
                
                prev_gray = gray
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(motion_intensities), np.array(timestamps)
    
    def compute_brightness_variance(self, video_path):
        """
        Compute brightness variance over time
        Sudden changes = visual interest
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps / self.sample_rate) if self.sample_rate < fps else 1
        
        brightness_values = []
        timestamps = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # Convert to LAB color space, get L channel (brightness)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                mean_brightness = np.mean(l_channel)
                
                brightness_values.append(mean_brightness)
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(brightness_values), np.array(timestamps)
    
    def detect_faces(self, video_path):
        """
        Detect faces in video (simple Haar Cascade)
        Face presence = engagement, personality
        """
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps / self.sample_rate) if self.sample_rate < fps else 1
        
        face_counts = []
        timestamps = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                face_counts.append(len(faces))
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        return np.array(face_counts), np.array(timestamps)
    
    def compute_visual_saliency(self, scene_changes, motion, brightness_variance):
        """
        Combine visual features into saliency score
        High saliency = visually interesting moment
        """
        # Normalize features
        motion_norm = (motion - np.mean(motion)) / (np.std(motion) + 1e-10)
        brightness_var = np.array([np.std(brightness_variance[max(0, i-5):i+6]) 
                                   for i in range(len(brightness_variance))])
        brightness_var_norm = (brightness_var - np.mean(brightness_var)) / (np.std(brightness_var) + 1e-10)
        
        # Weighted combination
        saliency = (
            0.4 * motion_norm +
            0.3 * brightness_var_norm +
            0.3 * scene_changes.astype(float)
        )
        
        return saliency
    
    def extract_all_features(self, video_path):
        """
        Extract all visual features
        
        Returns:
            dict with timestamps and feature arrays
        """
        # Get video info
        info = self.get_video_info(video_path)
        
        # Extract features
        scene_changes, sc_timestamps = self.detect_scene_changes(video_path)
        motion, motion_timestamps = self.compute_motion_intensity(video_path)
        brightness, brightness_timestamps = self.compute_brightness_variance(video_path)
        faces, face_timestamps = self.detect_faces(video_path)
        
        # Use shortest timestamp array as reference
        min_len = min(len(sc_timestamps), len(motion_timestamps), 
                     len(brightness_timestamps), len(face_timestamps))
        
        timestamps = sc_timestamps[:min_len]
        scene_changes = scene_changes[:min_len]
        motion = motion[:min_len]
        brightness = brightness[:min_len]
        faces = faces[:min_len]
        
        # Compute saliency
        saliency = self.compute_visual_saliency(scene_changes, motion, brightness)
        
        features = {
            'timestamps': timestamps.tolist(),
            'scene_changes': scene_changes.tolist(),
            'scene_change_count': int(np.sum(scene_changes)),
            'motion_intensity': motion.tolist(),
            'motion_mean': float(np.mean(motion)),
            'motion_std': float(np.std(motion)),
            'brightness': brightness.tolist(),
            'brightness_variance': float(np.var(brightness)),
            'face_count': faces.tolist(),
            'face_presence_ratio': float(np.mean(faces > 0)),
            'visual_saliency': saliency.tolist(),
            'video_info': info
        }
        
        return features


if __name__ == '__main__':
    # Test
    extractor = VisualFeatureExtractor(sample_rate=1.0)
    # features = extractor.extract_all_features('test_video.mp4')
    # print(f"Extracted {len(features['timestamps'])} time windows")
    # print(f"Scene changes: {features['scene_change_count']}")
    # print(f"Mean motion: {features['motion_mean']:.3f}")
    print("Visual feature extractor ready")
