"""
Quality Filter Module
Validates clip coherence and filters false positives
"""

import numpy as np
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from features.transcript import TranscriptExtractor


class ClipQualityFilter:
    """Filter and validate clip quality"""
    
    def __init__(self):
        self.transcript_extractor = TranscriptExtractor()
    
    def validate_coherence(self, clip, transcript):
        """
        Check if clip is semantically coherent
        
        Returns:
            (bool, str) - (is_coherent, reason)
        """
        start_time = clip['start']
        end_time = clip['end']
        
        # Get clip text
        clip_text = self.transcript_extractor.get_transcript_slice(
            transcript,
            start_time,
            end_time
        )
        
        if not clip_text or len(clip_text.strip()) == 0:
            return False, "No transcript content"
        
        # 1. Check sentence completeness
        is_complete = self.transcript_extractor.check_sentence_completeness(clip_text)
        if not is_complete:
            return False, "Incomplete sentences"
        
        # 2. Check for unresolved references
        has_unresolved = self.transcript_extractor.has_unresolved_references(clip_text)
        if has_unresolved:
            return False, "Unresolved references (requires context)"
        
        # 3. Check minimum word count
        word_count = len(clip_text.split())
        if word_count < 20:
            return False, f"Too short ({word_count} words)"
        
        # 4. Check semantic density (not too sparse)
        density = self.transcript_extractor.compute_semantic_density(clip_text, window_size=20)
        if density < 0.3:
            return False, "Low semantic density"
        
        return True, "Coherent"
    
    def check_silence_ratio(self, clip, audio_features, max_silence=0.3):
        """
        Check if clip has too much silence
        
        Returns:
            (bool, str) - (passes, reason)
        """
        start_time = clip['start']
        end_time = clip['end']
        
        # Get audio slice
        timestamps = np.array(audio_features['timestamps'])
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return True, "No audio data"
        
        silence_values = [audio_features['silence'][i] for i in indices]
        silence_ratio = np.mean(silence_values)
        
        if silence_ratio > max_silence:
            return False, f"Too much silence ({silence_ratio:.1%})"
        
        return True, f"Silence OK ({silence_ratio:.1%})"
    
    def check_audio_quality(self, clip, audio_features, min_energy=0.05):
        """
        Check audio quality metrics
        
        Returns:
            (bool, str) - (passes, reason)
        """
        start_time = clip['start']
        end_time = clip['end']
        
        timestamps = np.array(audio_features['timestamps'])
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return True, "No audio data"
        
        # Check minimum energy
        energy_values = [audio_features['energy'][i] for i in indices]
        mean_energy = np.mean(energy_values)
        
        if mean_energy < min_energy:
            return False, f"Low audio energy ({mean_energy:.3f})"
        
        return True, f"Audio quality OK"
    
    def check_visual_quality(self, clip, visual_features):
        """
        Check visual quality (optional, can skip if too slow)
        
        Returns:
            (bool, str) - (passes, reason)
        """
        # For now, always pass
        # In production, check for:
        # - Brightness consistency
        # - Not entirely black frames
        # - Reasonable motion
        
        return True, "Visual quality OK"
    
    def is_intro_outro(self, clip, video_duration, threshold=0.05):
        """
        Check if clip is from intro or outro (usually skip these)
        
        Args:
            threshold: Fraction of video duration (0.05 = first/last 5%)
        
        Returns:
            bool - True if intro/outro
        """
        start_time = clip['start']
        end_time = clip['end']
        
        intro_threshold = video_duration * threshold
        outro_threshold = video_duration * (1 - threshold)
        
        # Check if entirely in intro or outro region
        if end_time < intro_threshold:
            return True  # In intro
        
        if start_time > outro_threshold:
            return True  # In outro
        
        return False
    
    def check_duplicate_content(self, clip, existing_clips, similarity_threshold=0.8):
        """
        Check if clip is too similar to existing clips
        
        Args:
            clip: Candidate clip
            existing_clips: List of already selected clips
            similarity_threshold: Jaccard similarity threshold
        
        Returns:
            (bool, str) - (is_unique, reason)
        """
        if not existing_clips:
            return True, "First clip"
        
        clip_start = clip['start']
        clip_end = clip['end']
        
        for existing_clip in existing_clips:
            ex_start = existing_clip['start']
            ex_end = existing_clip['end']
            
            # Compute overlap
            overlap_start = max(clip_start, ex_start)
            overlap_end = min(clip_end, ex_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            clip_duration = clip_end - clip_start
            ex_duration = ex_end - ex_start
            
            # Jaccard similarity
            union_duration = clip_duration + ex_duration - overlap_duration
            similarity = overlap_duration / union_duration if union_duration > 0 else 0
            
            if similarity > similarity_threshold:
                return False, f"Too similar to existing clip ({similarity:.1%})"
        
        return True, "Unique content"
    
    def filter_clips(self, clips, audio_features, visual_features, transcript,
                    video_duration, max_clips=10, skip_intro_outro=True):
        """
        Apply all quality filters to clip candidates
        
        Args:
            clips: List of candidate clips (already scored)
            audio_features: Full audio features
            visual_features: Full visual features
            transcript: Full transcript
            video_duration: Total video duration
            max_clips: Maximum clips to return
            skip_intro_outro: Whether to skip intro/outro clips
        
        Returns:
            List of high-quality, filtered clips
        """
        filtered_clips = []
        
        for clip in clips:
            # Check intro/outro
            if skip_intro_outro and self.is_intro_outro(clip, video_duration):
                continue
            
            # Check coherence
            is_coherent, reason = self.validate_coherence(clip, transcript)
            if not is_coherent:
                clip['filter_reason'] = reason
                continue
            
            # Check silence
            passes_silence, reason = self.check_silence_ratio(clip, audio_features)
            if not passes_silence:
                clip['filter_reason'] = reason
                continue
            
            # Check audio quality
            passes_audio, reason = self.check_audio_quality(clip, audio_features)
            if not passes_audio:
                clip['filter_reason'] = reason
                continue
            
            # Check duplicates
            is_unique, reason = self.check_duplicate_content(clip, filtered_clips)
            if not is_unique:
                clip['filter_reason'] = reason
                continue
            
            # Passed all filters
            clip['filter_reason'] = 'Passed all filters'
            filtered_clips.append(clip)
            
            # Check if we have enough
            if len(filtered_clips) >= max_clips:
                break
        
        return filtered_clips
    
    def rank_clips(self, clips):
        """
        Final ranking of clips
        Combines viral score with other quality metrics
        
        Returns:
            Sorted list of clips (best first)
        """
        # For now, just sort by score
        # In production, can combine:
        # - Viral score
        # - Hook strength
        # - Coherence score
        # - Position in video (recency bias)
        
        ranked = sorted(clips, key=lambda x: x.get('score', 0), reverse=True)
        return ranked


if __name__ == '__main__':
    # Test
    filter = ClipQualityFilter()
    print("Quality filter ready")
