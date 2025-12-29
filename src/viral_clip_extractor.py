"""
Main Viral Clip Extraction Pipeline
Context-aware viral clip detection system
"""

import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from features.audio import AudioFeatureExtractor
from features.visual import VisualFeatureExtractor
from features.transcript import TranscriptExtractor
from models.context_encoder import VideoContextAnalyzer
from models.scorer import ViralClipScorer
from pipeline.boundaries import BoundaryRefiner
from pipeline.quality import ClipQualityFilter


class ViralClipExtractor:
    """Main pipeline for extracting viral clips from videos"""
    
    def __init__(self, window_size=0.5, sample_rate=1.0):
        """
        Args:
            window_size: Audio analysis window size (seconds)
            sample_rate: Visual analysis sample rate (frames per second)
        """
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor(sample_rate=sample_rate)
        self.transcript_extractor = TranscriptExtractor()
        self.context_analyzer = VideoContextAnalyzer()
        self.boundary_refiner = BoundaryRefiner()
        self.quality_filter = ClipQualityFilter()
        
        self.window_size = window_size
        self.sample_rate = sample_rate
    
    def generate_candidate_segments(self, duration, target_duration=60, overlap=0.5):
        """
        Generate candidate segments using sliding window
        
        Args:
            duration: Total video duration
            target_duration: Target clip length
            overlap: Overlap ratio (0.5 = 50% overlap)
        
        Returns:
            List of segment dicts with start, end, duration
        """
        step_size = int(target_duration * (1 - overlap))
        segments = []
        
        current_start = 0
        while current_start + target_duration <= duration:
            segments.append({
                'start': current_start,
                'end': current_start + target_duration,
                'duration': target_duration
            })
            current_start += step_size
        
        return segments
    
    def extract_clips(self, video_path, num_clips=10, target_duration=60,
                     video_metadata=None, progress_callback=None):
        """
        Main extraction pipeline
        
        Args:
            video_path: Path to input video
            num_clips: Number of clips to generate
            target_duration: Target clip duration (seconds)
            video_metadata: Optional dict with title, description
            progress_callback: Optional callback(stage, progress) for progress updates
        
        Returns:
            List of final clip dicts with metadata
        """
        start_time = time.time()
        
        def update_progress(stage, progress):
            if progress_callback:
                progress_callback(stage, progress)
        
        # ========================================
        # STAGE 1: Feature Extraction
        # ========================================
        update_progress("Extracting audio features", 0.1)
        audio_features = self.audio_extractor.extract_all_features(
            video_path,
            window_size=self.window_size
        )
        
        update_progress("Extracting visual features", 0.3)
        visual_features = self.visual_extractor.extract_all_features(video_path)
        
        update_progress("Extracting transcript", 0.5)
        # Extract audio for transcript
        audio_path = audio_features['audio_path']
        transcript_features = self.transcript_extractor.extract_all_features(audio_path)
        
        video_duration = visual_features['video_info']['duration']
        
        # ========================================
        # STAGE 2: Global Context Analysis
        # ========================================
        update_progress("Analyzing video context", 0.6)
        global_context = self.context_analyzer.build_global_context(
            video_path=video_path,
            audio_features=audio_features,
            visual_features=visual_features,
            transcript_features=transcript_features,
            video_metadata=video_metadata
        )
        
        domain = global_context['domain']
        
        # ========================================
        # STAGE 3: Segment Generation & Scoring
        # ========================================
        update_progress("Generating candidate segments", 0.65)
        candidate_segments = self.generate_candidate_segments(
            duration=video_duration,
            target_duration=target_duration,
            overlap=0.5
        )
        
        update_progress("Scoring segments", 0.7)
        scorer = ViralClipScorer(domain=domain)
        scored_segments = scorer.score_segments(
            candidate_segments,
            audio_features,
            visual_features,
            transcript_features,
            global_context
        )
        
        # Sort by score
        scored_segments = sorted(scored_segments, key=lambda x: x['score'], reverse=True)
        
        # Take top candidates (oversample for filtering)
        top_candidates = scored_segments[:num_clips * 3]
        
        # ========================================
        # STAGE 4: Boundary Refinement
        # ========================================
        update_progress("Refining clip boundaries", 0.8)
        refined_clips = []
        
        for segment in top_candidates:
            refined = self.boundary_refiner.refine_boundaries(
                segment,
                audio_features,
                visual_features,
                transcript_features['transcript'],
                min_duration=max(20, target_duration * 0.5),
                max_duration=min(120, target_duration * 1.5)
            )
            refined_clips.append(refined)
        
        # ========================================
        # STAGE 5: Quality Filtering & Ranking
        # ========================================
        update_progress("Filtering and ranking clips", 0.9)
        final_clips = self.quality_filter.filter_clips(
            refined_clips,
            audio_features,
            visual_features,
            transcript_features['transcript'],
            video_duration,
            max_clips=num_clips,
            skip_intro_outro=True
        )
        
        # Final ranking
        final_clips = self.quality_filter.rank_clips(final_clips)
        
        # Add metadata
        for i, clip in enumerate(final_clips):
            clip['index'] = i + 1
            clip['video_path'] = video_path
            clip['domain'] = domain
            clip['topics'] = global_context['topics']
            
            # Get clip text
            clip['text'] = self.transcript_extractor.get_transcript_slice(
                transcript_features['transcript'],
                clip['start'],
                clip['end']
            )
            
            # Generate reason
            clip['reason'] = self._generate_reason(clip, domain)
        
        update_progress("Complete", 1.0)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'clips': final_clips,
            'domain': domain,
            'topics': global_context['topics'],
            'video_duration': video_duration,
            'processing_time': elapsed_time,
            'stats': {
                'candidates_generated': len(candidate_segments),
                'candidates_scored': len(scored_segments),
                'candidates_refined': len(refined_clips),
                'final_clips': len(final_clips)
            }
        }
        
        return result
    
    def _generate_reason(self, clip, domain):
        """Generate human-readable reason for clip selection"""
        score = clip.get('score', 0)
        hook_score = clip.get('hook_score', 0)
        
        reasons = []
        
        # Score-based reasons
        if score > 0.8:
            reasons.append("Extremely high viral potential")
        elif score > 0.6:
            reasons.append("High engagement indicators")
        elif score > 0.4:
            reasons.append("Good viral signals")
        
        # Hook strength
        if hook_score > 0.7:
            reasons.append("strong opening hook")
        
        # Domain-specific
        domain_reasons = {
            'podcast': "engaging dialogue",
            'gaming': "intense action",
            'lecture': "key educational moment",
            'vlog': "interesting visual moment",
            'comedy': "comedic peak"
        }
        
        if domain in domain_reasons:
            reasons.append(domain_reasons[domain])
        
        # Fallback
        if not reasons:
            reasons.append("viral potential detected")
        
        return ", ".join(reasons).capitalize()


if __name__ == '__main__':
    # Test
    extractor = ViralClipExtractor()
    print("Viral clip extractor ready")
    print("To use:")
    print("  result = extractor.extract_clips('video.mp4', num_clips=10)")
    print("  clips = result['clips']")
