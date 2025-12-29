"""
Transcript Extraction Module
ASR transcription and text feature extraction
"""

import numpy as np
import subprocess
import json
import re
from pathlib import Path


class TranscriptExtractor:
    """Extract and process transcript from video audio"""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def simple_transcribe(self, audio_path):
        """
        Simple word-based mock transcription
        In production, replace with Whisper or other ASR
        """
        # For now, return mock transcript with timestamps
        # In real implementation: use faster-whisper or openai-whisper
        
        mock_transcript = {
            'text': "This is a mock transcript. In production, use Whisper for real transcription.",
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'This is a mock transcript.'},
                {'start': 5.0, 'end': 10.0, 'text': 'In production, use Whisper for real transcription.'}
            ],
            'words': [
                {'word': 'This', 'start': 0.0, 'end': 0.5},
                {'word': 'is', 'start': 0.5, 'end': 0.8},
                {'word': 'a', 'start': 0.8, 'end': 1.0},
                {'word': 'mock', 'start': 1.0, 'end': 1.5},
                {'word': 'transcript', 'start': 1.5, 'end': 2.5},
            ]
        }
        
        return mock_transcript
    
    def split_into_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def compute_semantic_density(self, text, window_size=50):
        """
        Compute semantic density (unique words per window)
        High density = information-rich content
        """
        words = text.lower().split()
        
        if len(words) < window_size:
            unique_ratio = len(set(words)) / max(len(words), 1)
            return unique_ratio
        
        densities = []
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window = words[i:i+window_size]
            unique_ratio = len(set(window)) / len(window)
            densities.append(unique_ratio)
        
        return np.mean(densities) if densities else 0.0
    
    def detect_questions(self, text):
        """Detect question words and question marks"""
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose']
        
        text_lower = text.lower()
        has_question_mark = '?' in text
        has_question_word = any(word in text_lower for word in question_words)
        
        return has_question_mark or has_question_word
    
    def detect_viral_keywords(self, text):
        """
        Detect viral trigger words
        These words correlate with engagement
        """
        viral_keywords = [
            'never', 'always', 'secret', 'shocking', 'insane', 'crazy',
            'unbelievable', 'amazing', 'incredible', 'wow', 'omg',
            'you won\'t believe', 'this is why', 'the reason',
            'here\'s how', 'watch this', 'check this out',
            'breaking', 'exclusive', 'revealed'
        ]
        
        text_lower = text.lower()
        count = sum(1 for keyword in viral_keywords if keyword in text_lower)
        
        return count
    
    def count_exclamations(self, text):
        """Count exclamation marks (excitement indicator)"""
        return text.count('!')
    
    def compute_dialogue_density(self, segments):
        """
        Measure dialogue density (speaker changes, back-and-forth)
        High dialogue = engaging conversation
        """
        if len(segments) < 2:
            return 0.0
        
        # Approximate: count short segments (likely dialogue)
        short_segments = sum(1 for seg in segments if (seg['end'] - seg['start']) < 3.0)
        dialogue_ratio = short_segments / len(segments)
        
        return dialogue_ratio
    
    def get_transcript_slice(self, transcript, start_time, end_time):
        """Extract transcript text for a time range"""
        if 'segments' not in transcript:
            return ""
        
        text_parts = []
        for segment in transcript['segments']:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check overlap
            if seg_end >= start_time and seg_start <= end_time:
                text_parts.append(segment['text'])
        
        return ' '.join(text_parts)
    
    def check_sentence_completeness(self, text):
        """
        Check if text contains complete sentences
        Returns True if starts and ends with complete sentences
        """
        if not text.strip():
            return False
        
        # Check if ends with sentence terminator
        ends_complete = text.strip()[-1] in '.!?'
        
        # Check if starts with capital letter
        starts_complete = text.strip()[0].isupper()
        
        return starts_complete and ends_complete
    
    def has_unresolved_references(self, text):
        """
        Detect pronouns or references that may require prior context
        e.g., "This is important" - what is "this"?
        """
        # Starting with connectives = incomplete
        connectives = ['because', 'therefore', 'however', 'so', 'but', 'and', 'then']
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        
        if first_word in connectives:
            return True
        
        # Dangling pronouns at start
        dangling_pronouns = ['this', 'that', 'these', 'those', 'it', 'they']
        starts_with_pronoun = first_word in dangling_pronouns
        
        return starts_with_pronoun
    
    def extract_all_features(self, audio_path):
        """
        Extract transcript and compute text features
        
        Returns:
            dict with transcript and features
        """
        # Get transcript (mock for now)
        transcript = self.simple_transcribe(audio_path)
        full_text = transcript['text']
        
        # Compute features
        features = {
            'transcript': transcript,
            'full_text': full_text,
            'sentences': self.split_into_sentences(full_text),
            'semantic_density': self.compute_semantic_density(full_text),
            'has_questions': self.detect_questions(full_text),
            'viral_keyword_count': self.detect_viral_keywords(full_text),
            'exclamation_count': self.count_exclamations(full_text),
            'dialogue_density': self.compute_dialogue_density(transcript['segments']),
            'is_complete': self.check_sentence_completeness(full_text),
            'word_count': len(full_text.split()),
        }
        
        return features


if __name__ == '__main__':
    # Test
    extractor = TranscriptExtractor()
    # features = extractor.extract_all_features('test_audio.wav')
    # print(f"Word count: {features['word_count']}")
    # print(f"Semantic density: {features['semantic_density']:.3f}")
    print("Transcript extractor ready")
