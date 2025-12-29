"""
Context Modeling Module
Builds global video understanding for context-aware clip extraction
"""

import numpy as np
from collections import Counter


class VideoContextAnalyzer:
    """Analyze global video context"""
    
    def __init__(self):
        pass
    
    def classify_video_domain(self, title="", description="", transcript_sample=""):
        """
        Classify video domain (podcast, gaming, lecture, etc.)
        Simple keyword-based for MVP - replace with ML classifier in production
        """
        content = (title + " " + description + " " + transcript_sample).lower()
        
        # Domain keywords
        domains = {
            'podcast': ['podcast', 'interview', 'conversation', 'talk', 'discussion'],
            'gaming': ['gameplay', 'gaming', 'stream', 'playthrough', 'walkthrough', 'game'],
            'lecture': ['lecture', 'tutorial', 'course', 'lesson', 'education', 'learn', 'teach'],
            'vlog': ['vlog', 'daily', 'routine', 'lifestyle', 'day in'],
            'comedy': ['funny', 'comedy', 'humor', 'laugh', 'joke', 'meme'],
        }
        
        scores = {}
        for domain, keywords in domains.items():
            score = sum(content.count(keyword) for keyword in keywords)
            scores[domain] = score
        
        # Return domain with highest score
        if max(scores.values()) == 0:
            return 'general'
        
        return max(scores, key=scores.get)
    
    def extract_topics(self, transcript_text, n_topics=5):
        """
        Extract main topics from transcript
        Simple word frequency for MVP - use BERTopic in production
        """
        # Stopwords to ignore
        stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'not', 'no', 'yes', 'can', 'will', 'would', 'should', 'could'
        ])
        
        # Tokenize and clean
        words = transcript_text.lower().split()
        words = [w.strip('.,!?;:') for w in words]
        words = [w for w in words if w and w not in stopwords and len(w) > 3]
        
        # Get top words as topics
        word_counts = Counter(words)
        topics = [word for word, count in word_counts.most_common(n_topics)]
        
        return topics
    
    def analyze_narrative_structure(self, transcript_segments, audio_features):
        """
        Analyze narrative structure (setup, buildup, climax, resolution)
        Based on emotion intensity progression
        """
        if not transcript_segments or 'emotion_intensity' not in audio_features:
            return {
                'setup': (0.0, 0.2),
                'buildup': (0.2, 0.6),
                'climax': (0.6, 0.7),
                'resolution': (0.7, 1.0)
            }
        
        emotion_timeline = np.array(audio_features['emotion_intensity'])
        
        # Find climax (max emotion)
        if len(emotion_timeline) == 0:
            max_idx = 0
        else:
            max_idx = np.argmax(emotion_timeline)
        
        total_len = len(emotion_timeline)
        max_ratio = max_idx / max(total_len, 1)
        
        # Define narrative regions as ratios
        structure = {
            'setup': (0.0, max_ratio * 0.4),
            'buildup': (max_ratio * 0.4, max_ratio * 0.9),
            'climax': (max_ratio * 0.9, min(max_ratio * 1.1, 1.0)),
            'resolution': (min(max_ratio * 1.1, 1.0), 1.0)
        }
        
        return structure
    
    def compute_topic_timeline(self, transcript_segments, topics, window_size=10):
        """
        Map topics over time
        Shows which topics appear when in the video
        """
        topic_timeline = []
        
        for i, segment in enumerate(transcript_segments):
            text = segment.get('text', '').lower()
            segment_topics = [topic for topic in topics if topic in text]
            topic_timeline.append(segment_topics)
        
        return topic_timeline
    
    def build_global_context(self, video_path, audio_features, visual_features, 
                            transcript_features, video_metadata=None):
        """
        Build comprehensive global video context
        
        Args:
            video_path: Path to video
            audio_features: Dict from AudioFeatureExtractor
            visual_features: Dict from VisualFeatureExtractor
            transcript_features: Dict from TranscriptExtractor
            video_metadata: Optional dict with title, description
        
        Returns:
            Global context dictionary
        """
        metadata = video_metadata or {'title': '', 'description': ''}
        
        # Classify domain
        domain = self.classify_video_domain(
            title=metadata.get('title', ''),
            description=metadata.get('description', ''),
            transcript_sample=transcript_features['full_text'][:2000]
        )
        
        # Extract topics
        topics = self.extract_topics(transcript_features['full_text'])
        
        # Analyze narrative
        narrative_structure = self.analyze_narrative_structure(
            transcript_features['transcript']['segments'],
            audio_features
        )
        
        # Topic timeline
        topic_timeline = self.compute_topic_timeline(
            transcript_features['transcript']['segments'],
            topics
        )
        
        # Compute global statistics
        global_stats = {
            'avg_energy': audio_features.get('energy_mean', 0),
            'energy_variance': audio_features.get('energy_std', 0) ** 2,
            'avg_motion': visual_features.get('motion_mean', 0),
            'scene_change_rate': visual_features.get('scene_change_count', 0) / max(len(visual_features.get('timestamps', [1])), 1),
            'dialogue_density': transcript_features.get('dialogue_density', 0),
            'semantic_density': transcript_features.get('semantic_density', 0),
            'viral_keyword_density': transcript_features.get('viral_keyword_count', 0) / max(transcript_features.get('word_count', 1), 1),
        }
        
        # Build context
        context = {
            'video_path': video_path,
            'domain': domain,
            'topics': topics,
            'topic_timeline': topic_timeline,
            'narrative_structure': narrative_structure,
            'global_stats': global_stats,
            'duration': visual_features.get('video_info', {}).get('duration', 0),
            'metadata': metadata
        }
        
        return context
    
    def score_narrative_position(self, timestamp, duration, narrative_structure):
        """
        Score a timestamp based on its narrative position
        Climax moments score higher
        """
        ratio = timestamp / max(duration, 1)
        
        # Check which narrative stage
        if narrative_structure['climax'][0] <= ratio <= narrative_structure['climax'][1]:
            return 2.0  # High priority
        elif narrative_structure['buildup'][0] <= ratio <= narrative_structure['buildup'][1]:
            return 1.5  # Medium-high priority
        elif narrative_structure['resolution'][0] <= ratio <= narrative_structure['resolution'][1]:
            return 1.0  # Medium priority
        else:  # setup
            return 0.8  # Lower priority
    
    def calculate_topic_novelty(self, segment_text, global_topics, topic_timeline, segment_idx):
        """
        Calculate how novel/unique a segment's topic is
        First mentions of key topics score higher
        """
        segment_topics = [topic for topic in global_topics if topic in segment_text.lower()]
        
        if not segment_topics:
            return 0.0
        
        # Check if this is first or early mention
        novelty_score = 0.0
        for topic in segment_topics:
            # Find first mention
            for i, topics_at_time in enumerate(topic_timeline):
                if topic in topics_at_time:
                    if i == segment_idx:
                        # This is the first mention
                        novelty_score += 2.0
                    elif i < segment_idx and (segment_idx - i) < 5:
                        # Early mention (within 5 segments)
                        novelty_score += 1.0
                    break
        
        return novelty_score / len(segment_topics) if segment_topics else 0.0


if __name__ == '__main__':
    # Test
    analyzer = VideoContextAnalyzer()
    domain = analyzer.classify_video_domain(
        title="My Gaming Stream",
        description="Playing the new game"
    )
    print(f"Detected domain: {domain}")
    print("Context analyzer ready")
