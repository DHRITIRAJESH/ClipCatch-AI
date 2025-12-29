"""
Boundary Refinement Module
Intelligently adjusts clip boundaries for semantic completeness
"""

import numpy as np
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from features.transcript import TranscriptExtractor


class BoundaryRefiner:
    """Refine clip boundaries for completeness"""
    
    def __init__(self):
        self.transcript_extractor = TranscriptExtractor()
    
    def find_nearest_sentence_boundary(self, transcript, timestamp, direction='start'):
        """
        Find nearest sentence boundary near timestamp
        
        Args:
            transcript: Full transcript dict with segments
            timestamp: Target timestamp
            direction: 'start' or 'end' - which boundary to align
        
        Returns:
            Adjusted timestamp at sentence boundary
        """
        if 'segments' not in transcript:
            return timestamp
        
        segments = transcript['segments']
        
        # Find segment containing or near timestamp
        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            
            if direction == 'start':
                # Align to start of segment if close
                if abs(seg_start - timestamp) < 3.0:  # Within 3 seconds
                    return seg_start
            else:  # direction == 'end'
                # Align to end of segment if close
                if abs(seg_end - timestamp) < 3.0:
                    return seg_end
        
        return timestamp
    
    def check_hook_strength(self, start_time, audio_features, transcript, hook_duration=3.0):
        """
        Check if first few seconds are compelling (strong hook)
        
        Returns:
            float score (0-1)
        """
        hook_end = start_time + hook_duration
        
        # Get audio features for hook window
        timestamps = np.array(audio_features['timestamps'])
        mask = (timestamps >= start_time) & (timestamps < hook_end)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return 0.0
        
        hook_score = 0.0
        
        # 1. Audio energy
        hook_energy = np.mean([audio_features['energy'][i] for i in indices])
        global_energy = audio_features['energy_mean']
        if hook_energy > global_energy * 1.2:
            hook_score += 0.3
        
        # 2. No silence
        hook_silence = np.mean([audio_features['silence'][i] for i in indices])
        if hook_silence < 0.2:
            hook_score += 0.3
        
        # 3. Hook text (question, viral keywords)
        hook_text = self.transcript_extractor.get_transcript_slice(
            transcript,
            start_time,
            hook_end
        )
        
        if hook_text:
            has_question = self.transcript_extractor.detect_questions(hook_text)
            viral_keywords = self.transcript_extractor.detect_viral_keywords(hook_text)
            
            if has_question:
                hook_score += 0.2
            if viral_keywords > 0:
                hook_score += 0.2
        
        return min(hook_score, 1.0)
    
    def adjust_for_hook(self, start_time, audio_features, transcript, max_adjustment=10.0):
        """
        Try to improve hook by adjusting start time forward
        
        Returns:
            Best start time with strong hook
        """
        best_start = start_time
        best_hook_score = self.check_hook_strength(start_time, audio_features, transcript)
        
        # Try adjustments (1 second steps)
        for offset in range(1, int(max_adjustment) + 1):
            new_start = start_time + offset
            hook_score = self.check_hook_strength(new_start, audio_features, transcript)
            
            if hook_score > best_hook_score:
                best_hook_score = hook_score
                best_start = new_start
        
        return best_start, best_hook_score
    
    def check_emotional_closure(self, end_time, audio_features):
        """
        Check if clip ends at emotional closure (not mid-peak)
        
        Returns:
            True if good closure, False if mid-peak
        """
        timestamps = np.array(audio_features['timestamps'])
        
        # Check emotion intensity around end time
        window_size = 2.0  # seconds
        mask = (timestamps >= end_time - window_size) & (timestamps <= end_time + window_size)
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            return True
        
        # Get emotion intensity
        emotions = [audio_features['emotion_intensity'][i] for i in indices]
        
        # Check if we're at a peak
        mid_idx = len(emotions) // 2
        if mid_idx > 0 and mid_idx < len(emotions) - 1:
            is_peak = emotions[mid_idx] > emotions[mid_idx - 1] and \
                     emotions[mid_idx] > emotions[mid_idx + 1]
            
            return not is_peak  # Good closure if NOT at peak
        
        return True
    
    def extend_for_closure(self, end_time, audio_features, max_extension=5.0):
        """
        Extend clip to find emotional closure
        
        Returns:
            Extended end time
        """
        # Check if already has closure
        if self.check_emotional_closure(end_time, audio_features):
            return end_time
        
        # Try extending
        for offset in range(1, int(max_extension) + 1):
            new_end = end_time + offset
            if self.check_emotional_closure(new_end, audio_features):
                return new_end
        
        return end_time
    
    def refine_boundaries(self, segment, audio_features, visual_features,
                         transcript, min_duration=30, max_duration=90):
        """
        Refine segment boundaries for completeness
        
        Args:
            segment: Dict with 'start', 'end'
            audio_features: Full audio features
            visual_features: Full visual features
            transcript: Full transcript
            min_duration: Minimum clip duration (seconds)
            max_duration: Maximum clip duration (seconds)
        
        Returns:
            Refined segment dict
        """
        start_time = segment['start']
        end_time = segment['end']
        
        # 1. Align to sentence boundaries
        start_time = self.find_nearest_sentence_boundary(
            transcript,
            start_time,
            direction='start'
        )
        
        end_time = self.find_nearest_sentence_boundary(
            transcript,
            end_time,
            direction='end'
        )
        
        # 2. Optimize hook (first 3 seconds)
        start_time, hook_score = self.adjust_for_hook(
            start_time,
            audio_features,
            transcript
        )
        
        # 3. Ensure emotional closure
        end_time = self.extend_for_closure(
            end_time,
            audio_features,
            max_extension=5.0
        )
        
        # 4. Enforce duration constraints
        duration = end_time - start_time
        
        if duration < min_duration:
            # Extend end
            end_time = start_time + min_duration
        elif duration > max_duration:
            # Shrink from end, re-align to sentence
            end_time = start_time + max_duration
            end_time = self.find_nearest_sentence_boundary(
                transcript,
                end_time,
                direction='end'
            )
        
        # Final duration
        final_duration = end_time - start_time
        
        refined_segment = {
            'start': start_time,
            'end': end_time,
            'duration': final_duration,
            'hook_score': hook_score,
            'original_start': segment['start'],
            'original_end': segment['end']
        }
        
        # Preserve score if it exists
        if 'score' in segment:
            refined_segment['score'] = segment['score']
        
        return refined_segment


if __name__ == '__main__':
    # Test
    refiner = BoundaryRefiner()
    print("Boundary refiner ready")
