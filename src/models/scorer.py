"""
Viral Clip Scorer
Context-aware scoring of video segments for viral potential
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from features.transcript import TranscriptExtractor


class ViralClipScorer:
    """Score video segments based on viral potential"""
    
    def __init__(self, domain='general'):
        """
        Args:
            domain: Video domain (podcast, gaming, lecture, vlog, comedy, general)
        """
        self.domain = domain
        self.transcript_extractor = TranscriptExtractor()
        
        # Domain-specific weights
        self.domain_weights = {
            'podcast': {
                'audio_energy': 0.15,
                'audio_variance': 0.10,
                'emotion_intensity': 0.20,
                'semantic_density': 0.15,
                'dialogue_density': 0.15,
                'viral_keywords': 0.10,
                'topic_novelty': 0.10,
                'narrative_position': 0.05
            },
            'gaming': {
                'audio_energy': 0.20,
                'audio_variance': 0.15,
                'emotion_intensity': 0.15,
                'visual_motion': 0.20,
                'visual_saliency': 0.15,
                'viral_keywords': 0.10,
                'narrative_position': 0.05
            },
            'lecture': {
                'semantic_density': 0.25,
                'topic_novelty': 0.20,
                'viral_keywords': 0.15,
                'visual_saliency': 0.15,
                'narrative_position': 0.15,
                'audio_energy': 0.10
            },
            'vlog': {
                'visual_motion': 0.20,
                'visual_saliency': 0.20,
                'emotion_intensity': 0.15,
                'viral_keywords': 0.15,
                'audio_energy': 0.15,
                'narrative_position': 0.15
            },
            'comedy': {
                'emotion_intensity': 0.25,
                'audio_variance': 0.20,
                'viral_keywords': 0.20,
                'semantic_density': 0.15,
                'narrative_position': 0.10,
                'audio_energy': 0.10
            },
            'general': {
                'audio_energy': 0.15,
                'emotion_intensity': 0.15,
                'visual_saliency': 0.15,
                'semantic_density': 0.15,
                'viral_keywords': 0.15,
                'topic_novelty': 0.10,
                'narrative_position': 0.15
            }
        }
    
    def get_audio_slice(self, audio_features, start_time, end_time):
        """Extract audio features for time range"""
        timestamps = np.array(audio_features['timestamps'])
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        
        slice_features = {
            'energy': [audio_features['energy'][i] for i in indices],
            'emotion_intensity': [audio_features['emotion_intensity'][i] for i in indices],
            'pitch': [audio_features['pitch'][i] for i in indices],
            'silence': [audio_features['silence'][i] for i in indices],
        }
        
        return slice_features
    
    def get_visual_slice(self, visual_features, start_time, end_time):
        """Extract visual features for time range"""
        timestamps = np.array(visual_features['timestamps'])
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        
        slice_features = {
            'motion_intensity': [visual_features['motion_intensity'][i] for i in indices],
            'visual_saliency': [visual_features['visual_saliency'][i] for i in indices],
            'scene_changes': [visual_features['scene_changes'][i] for i in indices],
        }
        
        return slice_features
    
    def score_segment(self, segment, audio_features, visual_features, 
                     transcript_features, global_context):
        """
        Score a single segment for viral potential
        
        Args:
            segment: Dict with 'start', 'end', 'duration'
            audio_features: Full video audio features
            visual_features: Full video visual features
            transcript_features: Full video transcript features
            global_context: Global video context from VideoContextAnalyzer
        
        Returns:
            float score (0-1)
        """
        start_time = segment['start']
        end_time = segment['end']
        duration = segment['duration']
        
        # Get slice features
        audio_slice = self.get_audio_slice(audio_features, start_time, end_time)
        visual_slice = self.get_visual_slice(visual_features, start_time, end_time)
        
        # Get transcript for this segment
        segment_text = self.transcript_extractor.get_transcript_slice(
            transcript_features['transcript'],
            start_time,
            end_time
        )
        
        # Calculate individual feature scores (0-1 normalized)
        feature_scores = {}
        
        # 1. Audio Energy
        if audio_slice and 'energy' in audio_slice:
            energy_mean = np.mean(audio_slice['energy'])
            global_energy = global_context['global_stats']['avg_energy']
            feature_scores['audio_energy'] = min(energy_mean / max(global_energy * 1.5, 0.01), 1.0)
        else:
            feature_scores['audio_energy'] = 0.0
        
        # 2. Audio Variance (excitement changes)
        if audio_slice and 'energy' in audio_slice:
            energy_std = np.std(audio_slice['energy'])
            global_variance = global_context['global_stats']['energy_variance']
            feature_scores['audio_variance'] = min(energy_std / max(np.sqrt(global_variance) * 1.5, 0.01), 1.0)
        else:
            feature_scores['audio_variance'] = 0.0
        
        # 3. Emotion Intensity
        if audio_slice and 'emotion_intensity' in audio_slice:
            emotion_mean = np.mean(audio_slice['emotion_intensity'])
            # Emotion is already normalized (-2 to +2 range)
            feature_scores['emotion_intensity'] = min(max(emotion_mean / 2.0, 0), 1.0)
        else:
            feature_scores['emotion_intensity'] = 0.0
        
        # 4. Visual Motion
        if visual_slice and 'motion_intensity' in visual_slice:
            motion_mean = np.mean(visual_slice['motion_intensity'])
            global_motion = global_context['global_stats']['avg_motion']
            feature_scores['visual_motion'] = min(motion_mean / max(global_motion * 1.5, 0.01), 1.0)
        else:
            feature_scores['visual_motion'] = 0.0
        
        # 5. Visual Saliency
        if visual_slice and 'visual_saliency' in visual_slice:
            saliency_mean = np.mean(visual_slice['visual_saliency'])
            # Saliency is already normalized
            feature_scores['visual_saliency'] = min(max(saliency_mean / 2.0, 0), 1.0)
        else:
            feature_scores['visual_saliency'] = 0.0
        
        # 6. Semantic Density
        if segment_text:
            semantic_density = self.transcript_extractor.compute_semantic_density(segment_text)
            global_density = global_context['global_stats']['semantic_density']
            feature_scores['semantic_density'] = min(semantic_density / max(global_density * 1.3, 0.01), 1.0)
        else:
            feature_scores['semantic_density'] = 0.0
        
        # 7. Viral Keywords
        if segment_text:
            keyword_count = self.transcript_extractor.detect_viral_keywords(segment_text)
            word_count = len(segment_text.split())
            keyword_density = keyword_count / max(word_count, 1)
            feature_scores['viral_keywords'] = min(keyword_density * 10, 1.0)  # Scale up
        else:
            feature_scores['viral_keywords'] = 0.0
        
        # 8. Dialogue Density (for podcasts)
        if 'dialogue_density' in transcript_features:
            # Check if segment has dialogue patterns
            feature_scores['dialogue_density'] = transcript_features['dialogue_density']
        else:
            feature_scores['dialogue_density'] = 0.0
        
        # 9. Topic Novelty
        if segment_text and 'topics' in global_context:
            # Simplified topic novelty - check if key topics appear
            segment_has_topic = any(topic in segment_text.lower() 
                                   for topic in global_context['topics'])
            feature_scores['topic_novelty'] = 1.0 if segment_has_topic else 0.3
        else:
            feature_scores['topic_novelty'] = 0.0
        
        # 10. Narrative Position
        from models.context_encoder import VideoContextAnalyzer
        analyzer = VideoContextAnalyzer()
        narrative_score = analyzer.score_narrative_position(
            start_time,
            global_context['duration'],
            global_context['narrative_structure']
        )
        feature_scores['narrative_position'] = min(narrative_score / 2.0, 1.0)
        
        # 11. Silence Penalty
        if audio_slice and 'silence' in audio_slice:
            silence_ratio = np.mean(audio_slice['silence'])
            silence_penalty = 1.0 - (silence_ratio * 0.5)  # Max 50% penalty
        else:
            silence_penalty = 1.0
        
        # Get domain-specific weights
        weights = self.domain_weights.get(self.domain, self.domain_weights['general'])
        
        # Weighted sum
        final_score = 0.0
        for feature, score in feature_scores.items():
            weight = weights.get(feature, 0.0)
            final_score += weight * score
        
        # Apply silence penalty
        final_score *= silence_penalty
        
        # Clip to [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score
    
    def score_segments(self, segments, audio_features, visual_features, 
                      transcript_features, global_context):
        """
        Score multiple segments
        
        Returns:
            List of (segment, score) tuples
        """
        scored_segments = []
        
        for segment in segments:
            score = self.score_segment(
                segment,
                audio_features,
                visual_features,
                transcript_features,
                global_context
            )
            
            scored_segments.append({
                **segment,
                'score': score
            })
        
        return scored_segments


if __name__ == '__main__':
    # Test
    scorer = ViralClipScorer(domain='podcast')
    print(f"Scorer initialized for domain: {scorer.domain}")
    print("Viral clip scorer ready")
