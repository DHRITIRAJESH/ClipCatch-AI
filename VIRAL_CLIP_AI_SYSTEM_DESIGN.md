# Context-Aware Viral Clip Extraction System

## Engineering Design & Implementation Guide

---

## 1️⃣ SYSTEM ARCHITECTURE

### High-Level Pipeline

```
RAW VIDEO
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Multi-Modal Feature Extraction (Parallel)     │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│ │   Audio     │  │    Visual    │  │  Transcription │ │
│ │ Extraction  │  │  Extraction  │  │   (ASR)        │ │
│ └─────────────┘  └──────────────┘  └────────────────┘ │
│      ↓                  ↓                   ↓          │
│  [Emotion,         [Scenes,           [Text,          │
│   Pitch,           Faces,             Embeddings,     │
│   Energy]          Motion]            Topics]         │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Global Context Modeling                       │
├─────────────────────────────────────────────────────────┤
│ • Video Classification (Domain Detection)              │
│ • Narrative Structure Analysis                         │
│ • Topic Segmentation                                    │
│ • Emotion Arc Mapping                                   │
│ • Speaker Diarization                                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: Context-Conditioned Segment Scoring           │
├─────────────────────────────────────────────────────────┤
│ • Sliding Window over timestamps                       │
│ • Per-segment viral potential score                    │
│ • Conditioned on global video context                  │
│ • Domain-specific scoring weights                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 4: Intelligent Boundary Detection                │
├─────────────────────────────────────────────────────────┤
│ • Semantic boundary refinement                         │
│ • Sentence completion detection                        │
│ • Emotional arc closure                                │
│ • Dynamic clip length optimization                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 5: Post-Processing & Quality Filtering           │
├─────────────────────────────────────────────────────────┤
│ • Overlap removal                                      │
│ • Coherence validation                                 │
│ • Hook detection (first 3 seconds)                     │
│ • Ranking & deduplication                              │
└─────────────────────────────────────────────────────────┘
    ↓
FINAL VIRAL CLIPS (with metadata)
```

### Module Breakdown

**Module 1: Feature Extractors**

- **Input**: Raw video file
- **Output**: Time-aligned multi-modal features
- **Components**: Audio processor, visual processor, ASR engine

**Module 2: Context Analyzer**

- **Input**: Multi-modal features
- **Output**: Global video understanding (structure, topics, domain)
- **Components**: Video classifier, topic modeler, narrative analyzer

**Module 3: Segment Scorer**

- **Input**: Features + global context
- **Output**: Scored candidate segments
- **Components**: Context encoder, scoring model

**Module 4: Boundary Refiner**

- **Input**: Raw segment boundaries + transcript
- **Output**: Semantically coherent clip boundaries
- **Components**: Sentence boundary detector, emotional closure analyzer

**Module 5: Quality Filter**

- **Input**: Candidate clips
- **Output**: Final ranked clips
- **Components**: Coherence validator, hook detector, ranker

---

## 2️⃣ MODEL CHOICES (JUSTIFIED)

### ASR Model: **Faster Whisper (Large-v3)**

**Why:**

- SOTA accuracy across domains (95%+ WER on podcasts)
- Fast inference with CTranslate2 backend (4-6x faster than baseline)
- Word-level timestamps crucial for boundary detection
- Robust to accents, background noise, multiple speakers
- Open-source, self-hostable

**Alternative**: AssemblyAI API (if cloud is acceptable, provides speaker diarization out-of-box)

**Implementation:**

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    audio_path,
    word_timestamps=True,
    vad_filter=True,  # Voice activity detection
    vad_parameters={"threshold": 0.5}
)
```

---

### Text Embedding: **all-MiniLM-L6-v2** (MVP) → **instructor-xl** (Production)

**Why:**

- **MiniLM**: Fast (384-dim), good enough for semantic similarity, retrieval
- **Instructor**: Task-specific embeddings, better understanding of "viral" semantics
- Both compatible with Sentence Transformers

**Usage:**

- Embed transcript chunks (5-10 word windows)
- Compute semantic similarity for coherence checking
- Topic clustering via UMAP + HDBSCAN

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks, convert_to_tensor=True)
```

---

### Audio Analysis: **librosa + pyAudioAnalysis**

**Why:**

- **librosa**: Standard for audio feature extraction (spectrograms, MFCCs, pitch)
- **pyAudioAnalysis**: Pre-built classifiers for emotion detection
- Lightweight, no GPU needed

**Features Extracted:**

- Energy (RMS) - detects excitement
- Pitch (F0) - vocal intensity
- Zero-crossing rate - speech vs silence
- Spectral centroid - tone brightness
- MFCC - emotion classification

```python
import librosa
import numpy as np

y, sr = librosa.load(audio_path, sr=16000)
energy = librosa.feature.rms(y=y)[0]
pitch = librosa.yin(y, fmin=50, fmax=400)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

---

### Visual Understanding: **CLIP (ViT-B/32)** + **OpenCV for Scene Detection**

**Why:**

- **CLIP**: Zero-shot image understanding, can detect "viral" visual patterns
- **OpenCV**: Fast scene change detection (necessary for clip boundaries)
- Lightweight compared to video transformers (TimeSformer, VideoMAE)

**Features Extracted:**

- Scene changes (cuts, transitions)
- Frame-level CLIP embeddings for visual similarity
- Face detection (OpenCV Haar Cascades or MediaPipe)
- Motion intensity (optical flow)

```python
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Scene detection
cap = cv2.VideoCapture(video_path)
prev_frame = None
scene_changes = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if prev_frame is not None:
        diff = cv2.absdiff(frame, prev_frame)
        if np.mean(diff) > threshold:
            scene_changes.append(frame_idx)

    prev_frame = frame
```

---

### Context Modeling: **Hierarchical Transformer (Custom)**

**Why:**

- Need to capture long-range dependencies (entire video)
- Hierarchical structure: sentence → paragraph → section → video
- Standard transformers (BERT, GPT) have limited context (512-4096 tokens)

**Architecture:**

```
Input: [Transcript embeddings + Audio features + Visual features] (T x D)
    ↓
Layer 1: Local Attention (segment-level, window=50 frames)
    ↓
Layer 2: Pooling (segment representations)
    ↓
Layer 3: Global Attention (cross-segment relationships)
    ↓
Output: Global video embedding (D-dim)
```

**Implementation Stack:**

- PyTorch
- HuggingFace Transformers (base architecture)
- Custom attention masks for hierarchical processing

**Alternative (Simpler MVP)**: Use **Longformer** or **BigBird** directly

---

### Clip Scoring Model: **XGBoost** (MVP) → **Neural Ranker** (Production)

**Why XGBoost (MVP):**

- Handles heterogeneous features (text, audio, visual, metadata)
- Fast training, interpretable (feature importance)
- Works well with limited labeled data
- Proven in ranking tasks (Kaggle, search engines)

**Why Neural Ranker (Production):**

- Better at learning complex interactions
- Can jointly optimize with context encoder
- End-to-end differentiable

**Features for Scoring:**

- Audio energy variance (excitement)
- Transcript semantic density (information content)
- Visual saliency (interesting frames)
- Topic novelty (first mention of key topics)
- Emotional peak detection
- Speaker engagement (multiple speakers in dialogue)
- Recency bias (later clips often better in podcasts)

```python
import xgboost as xgb

# Feature vector per segment
features = [
    audio_energy_mean,
    audio_energy_std,
    pitch_variance,
    semantic_density,
    visual_saliency_score,
    topic_novelty,
    emotion_peak_intensity,
    speaker_count,
    position_in_video  # normalized
]

model = xgb.XGBRanker(objective='rank:pairwise')
model.fit(X_train, y_train, group=group_sizes)
scores = model.predict(X_test)
```

---

### Domain Classifier: **DistilBERT** fine-tuned on video metadata

**Why:**

- Fast inference (66M params vs 110M for BERT)
- Good transfer learning with limited data
- Can use video title + description + first 2 minutes of transcript

**Classes:**

- Podcast
- Interview
- Lecture/Educational
- Gaming
- Vlog
- Comedy/Entertainment

**Training Data:**

- YouTube metadata (titles, tags, categories)
- Scraped from domain-specific channels

---

## 3️⃣ ALGORITHMIC FLOW

### Complete Pipeline (Pseudo-Code)

```python
def extract_viral_clips(video_path, num_clips=10, target_duration=60):
    """
    Main pipeline for viral clip extraction.

    Args:
        video_path: Path to input video
        num_clips: Number of clips to generate
        target_duration: Target clip length in seconds

    Returns:
        List of ClipMetadata objects with start/end times, scores
    """

    # ========================================
    # STAGE 1: Feature Extraction (Parallel)
    # ========================================

    # Extract audio
    audio_path = extract_audio(video_path)

    # Run in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Audio features
        future_audio = executor.submit(extract_audio_features, audio_path)

        # Transcript
        future_transcript = executor.submit(transcribe_audio, audio_path)

        # Visual features
        future_visual = executor.submit(extract_visual_features, video_path)

    audio_features = future_audio.result()
    transcript = future_transcript.result()
    visual_features = future_visual.result()

    # ========================================
    # STAGE 2: Global Context Modeling
    # ========================================

    # Classify video domain
    domain = classify_video_domain(
        title=video_metadata['title'],
        description=video_metadata['description'],
        transcript_sample=transcript[:2000]  # first 2 min
    )

    # Extract topics and narrative structure
    topics = extract_topics(transcript, method='BERTopic')
    narrative_structure = analyze_narrative_structure(transcript, audio_features)

    # Build global video representation
    global_context = build_global_context(
        transcript=transcript,
        audio_features=audio_features,
        visual_features=visual_features,
        topics=topics,
        narrative_structure=narrative_structure,
        domain=domain
    )

    # ========================================
    # STAGE 3: Segment Scoring
    # ========================================

    # Generate candidate segments (sliding window)
    candidate_segments = []
    step_size = target_duration // 2  # 50% overlap

    for start_time in range(0, video_duration - target_duration, step_size):
        end_time = start_time + target_duration

        segment = {
            'start': start_time,
            'end': end_time,
            'transcript': get_transcript_slice(transcript, start_time, end_time),
            'audio': get_audio_slice(audio_features, start_time, end_time),
            'visual': get_visual_slice(visual_features, start_time, end_time)
        }

        candidate_segments.append(segment)

    # Score each segment with global context
    scored_segments = []
    for segment in candidate_segments:
        score = score_segment(
            segment=segment,
            global_context=global_context,
            domain=domain
        )
        segment['score'] = score
        scored_segments.append(segment)

    # ========================================
    # STAGE 4: Boundary Refinement
    # ========================================

    # Select top segments
    top_segments = sorted(scored_segments, key=lambda x: x['score'], reverse=True)
    top_segments = top_segments[:num_clips * 2]  # Oversample for filtering

    # Refine boundaries
    refined_clips = []
    for segment in top_segments:
        refined_clip = refine_clip_boundaries(
            segment=segment,
            transcript=transcript,
            audio_features=audio_features,
            min_duration=30,
            max_duration=90
        )
        refined_clips.append(refined_clip)

    # ========================================
    # STAGE 5: Quality Filtering & Ranking
    # ========================================

    # Remove overlaps (keep higher scored)
    non_overlapping_clips = remove_overlapping_clips(refined_clips)

    # Validate coherence
    coherent_clips = [
        clip for clip in non_overlapping_clips
        if validate_coherence(clip, transcript)
    ]

    # Check for strong hooks (first 3 seconds)
    clips_with_hooks = [
        clip for clip in coherent_clips
        if has_strong_hook(clip, audio_features, transcript)
    ]

    # Final ranking (re-score with additional features)
    final_clips = rank_clips(
        clips_with_hooks,
        global_context=global_context,
        num_clips=num_clips
    )

    return final_clips


# ========================================
# Key Subroutines
# ========================================

def score_segment(segment, global_context, domain):
    """Context-conditioned segment scoring"""

    features = []

    # 1. Audio engagement
    audio = segment['audio']
    features.append(np.mean(audio['energy']))
    features.append(np.std(audio['energy']))
    features.append(np.max(audio['pitch_variance']))

    # 2. Semantic density
    text = segment['transcript']
    embeddings = embed_text(text)
    semantic_density = calculate_semantic_density(embeddings)
    features.append(semantic_density)

    # 3. Visual interest
    visual = segment['visual']
    features.append(np.mean(visual['scene_change_rate']))
    features.append(np.mean(visual['motion_intensity']))

    # 4. Topic novelty (vs. global context)
    segment_topics = extract_topics([text], method='fast')
    topic_novelty = calculate_topic_novelty(
        segment_topics,
        global_context['topics']
    )
    features.append(topic_novelty)

    # 5. Emotional peak
    emotion_score = detect_emotional_peak(
        audio['emotion_timeline'],
        segment['start'],
        segment['end']
    )
    features.append(emotion_score)

    # 6. Narrative position
    narrative_score = score_narrative_position(
        segment['start'],
        global_context['narrative_structure']
    )
    features.append(narrative_score)

    # 7. Domain-specific features
    if domain == 'podcast':
        features.append(count_speakers_in_segment(segment))
        features.append(dialogue_density(segment))
    elif domain == 'gaming':
        features.append(action_intensity(segment))

    # 8. Context similarity
    segment_embedding = encode_segment(segment)
    context_similarity = cosine_similarity(
        segment_embedding,
        global_context['embedding']
    )
    features.append(context_similarity)

    # Combine features (weighted or learned)
    score = scoring_model.predict([features])[0]
    return score


def refine_clip_boundaries(segment, transcript, audio_features,
                           min_duration=30, max_duration=90):
    """Intelligent boundary detection"""

    start_time = segment['start']
    end_time = segment['end']

    # 1. Find sentence boundaries near start/end
    sentences = get_sentences_with_timestamps(transcript)

    # Adjust start to beginning of sentence
    for sent in sentences:
        if abs(sent['start'] - start_time) < 3:  # within 3 seconds
            start_time = sent['start']
            break

    # Adjust end to end of sentence
    for sent in reversed(sentences):
        if abs(sent['end'] - end_time) < 3:
            end_time = sent['end']
            break

    # 2. Check for emotional closure
    # Don't cut during emotional peak
    emotion_timeline = audio_features['emotion_timeline']

    while is_emotional_peak(emotion_timeline, end_time):
        # Extend clip to capture full emotional arc
        end_time += 2
        if end_time - start_time > max_duration:
            break

    # 3. Ensure minimum hook duration
    # First 3 seconds should be compelling
    hook_score = score_hook(
        start_time,
        start_time + 3,
        audio_features,
        transcript
    )

    if hook_score < HOOK_THRESHOLD:
        # Try adjusting start time forward
        for offset in range(1, 10):
            new_start = start_time + offset
            new_hook_score = score_hook(
                new_start,
                new_start + 3,
                audio_features,
                transcript
            )
            if new_hook_score > hook_score:
                start_time = new_start
                break

    # 4. Ensure duration constraints
    duration = end_time - start_time
    if duration < min_duration:
        # Try extending end
        end_time = start_time + min_duration
    elif duration > max_duration:
        # Try shrinking from end
        end_time = start_time + max_duration
        # Re-adjust to sentence boundary
        for sent in reversed(sentences):
            if sent['end'] <= end_time:
                end_time = sent['end']
                break

    return {
        'start': start_time,
        'end': end_time,
        'duration': end_time - start_time,
        'score': segment['score']
    }


def validate_coherence(clip, transcript):
    """Check if clip is semantically coherent"""

    # 1. Must contain complete sentences
    clip_text = get_transcript_slice(transcript, clip['start'], clip['end'])
    sentences = split_into_sentences(clip_text)

    if len(sentences) == 0:
        return False

    # Check first/last sentences are complete
    if not is_complete_sentence(sentences[0]) or \
       not is_complete_sentence(sentences[-1]):
        return False

    # 2. Semantic coherence check
    # All sentences should be topically related
    embeddings = embed_text(sentences)
    similarities = pairwise_cosine_similarity(embeddings)

    avg_similarity = np.mean(similarities)
    if avg_similarity < 0.6:  # Threshold
        return False

    # 3. No abrupt topic changes
    # Check if clip spans topic boundaries
    topics = extract_topics(sentences, method='fast')
    if len(set(topics)) > 2:  # More than 2 topics = incoherent
        return False

    return True


def has_strong_hook(clip, audio_features, transcript):
    """Check if first 3 seconds hook viewer"""

    hook_start = clip['start']
    hook_end = clip['start'] + 3

    # Get hook content
    hook_text = get_transcript_slice(transcript, hook_start, hook_end)
    hook_audio = get_audio_slice(audio_features, hook_start, hook_end)

    score = 0

    # 1. Audio energy
    if np.mean(hook_audio['energy']) > 0.7:
        score += 1

    # 2. Question or provocative statement
    if '?' in hook_text or any(word in hook_text.lower() for word in
                                ['never', 'always', 'secret', 'shocking',
                                 'you won\'t believe', 'insane']):
        score += 1

    # 3. No silence
    silence_ratio = calculate_silence_ratio(hook_audio)
    if silence_ratio < 0.2:
        score += 1

    # 4. Emotional intensity
    if hook_audio['emotion_intensity'] > 0.6:
        score += 1

    return score >= 2  # At least 2 criteria met
```

---

## 4️⃣ TRAINING STRATEGY

### Data Sources

**Primary:**

1. **Labeled Dataset (Human Annotations)**

   - Source: YouTube videos with high engagement
   - Metrics: Views, likes, shares, comments
   - Clips: Extracted by humans or existing viral clips
   - Volume: 5,000-10,000 videos → 50,000+ clips

2. **Weak Supervision (Automatic Labels)**

   - Source: Existing viral clips on TikTok, Instagram Reels
   - Use timestamps in video descriptions
   - Volume: 100,000+ videos

3. **Synthetic Data (Negative Examples)**
   - Random segments from long videos
   - Low-engagement segments
   - Volume: 500,000+ segments

**Secondary:**

- Podcast transcripts (Spotify, Apple Podcasts)
- Educational videos (Khan Academy, MIT OpenCourseWare)
- Gaming streams (Twitch)

### Labeling Strategy

**Tier 1: Direct Labels (Expensive)**

- Hire annotators to watch videos and mark viral moments
- Guidelines:
  - "Would you share this 60s clip?"
  - Rate 1-5 on virality potential
- Cost: ~$0.50 per video (5-10 min watch time)
- Volume: 10,000 videos

**Tier 2: Weak Supervision (Cheap)**

- Engagement metrics as proxy labels
  - High engagement = positive samples
  - Low engagement = negative samples
- Formula: `score = (likes + shares*2 + comments*1.5) / views`
- Threshold: Top 10% = positive, Bottom 50% = negative

**Tier 3: Self-Supervision**

- Contrastive learning: Clip vs random segment
- Predict next segment (temporal coherence)
- Reconstruct global context from clip (context relevance)

**Data Augmentation:**

- Speed up/slow down audio (0.9x - 1.1x)
- Add background noise
- Crop/pad clip boundaries (±2 seconds)
- Mix speakers from different videos

### Training Pipeline

**Phase 1: Pre-training (Self-Supervised)**

```
Objective: Learn general video understanding
Dataset: 1M+ unlabeled videos
Tasks:
  - Masked Language Modeling (transcript)
  - Audio reconstruction
  - Frame prediction
  - Contrastive clip-context matching
Duration: 1-2 weeks on 8x A100 GPUs
```

**Phase 2: Fine-tuning (Supervised)**

```
Objective: Learn viral clip scoring
Dataset: 50K labeled clips + 500K weak labels
Loss: Pairwise ranking loss
  L = max(0, margin - score(positive) + score(negative))
Optimizer: AdamW, lr=1e-5, warmup=1000 steps
Batch size: 32 videos
Duration: 3-5 days on 4x A100 GPUs
```

**Phase 3: Domain Adaptation**

```
Objective: Specialize for each domain
Dataset: 10K clips per domain
Method: Fine-tune separate heads per domain
  - Shared encoder (frozen)
  - Domain-specific scoring layers
Duration: 1 day per domain
```

### Evaluation Metrics

**Online Metrics (A/B Testing)**

- **CTR**: Click-through rate on generated clips
- **Watch Time**: Average watch % (goal: >70%)
- **Engagement**: Likes, shares, comments on posted clips
- **Virality Score**: Combined metric of above

**Offline Metrics (Model Evaluation)**

- **NDCG@K**: Normalized Discounted Cumulative Gain
  - Measures ranking quality
  - K = 10 (top 10 clips)
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **Coherence Score**: Automated semantic validation
- **Boundary Accuracy**: How close to human-labeled boundaries

**Human Evaluation (Quality)**

- **Relevance**: Does clip make sense standalone? (1-5)
- **Hook Strength**: First 3 seconds compelling? (1-5)
- **Completeness**: Is the idea/story complete? (1-5)
- **Virality Potential**: Would you share? (1-5)

**Evaluation Protocol:**

```python
def evaluate_model(model, test_videos):
    results = []

    for video in test_videos:
        # Generate clips
        predicted_clips = model.extract_clips(video, num_clips=10)
        ground_truth_clips = video.human_annotations

        # Compute NDCG
        ndcg = calculate_ndcg(predicted_clips, ground_truth_clips)

        # Compute boundary accuracy
        boundary_acc = calculate_boundary_accuracy(
            predicted_clips,
            ground_truth_clips,
            tolerance=2.0  # seconds
        )

        # Coherence check
        coherence_scores = [
            validate_coherence(clip, video.transcript)
            for clip in predicted_clips
        ]
        coherence_rate = np.mean(coherence_scores)

        results.append({
            'video_id': video.id,
            'ndcg': ndcg,
            'boundary_acc': boundary_acc,
            'coherence_rate': coherence_rate
        })

    return {
        'mean_ndcg': np.mean([r['ndcg'] for r in results]),
        'mean_boundary_acc': np.mean([r['boundary_acc'] for r in results]),
        'mean_coherence': np.mean([r['coherence_rate'] for r in results])
    }
```

---

## 5️⃣ CLIP QUALITY CONTROLS

### Coherence Assurance

**1. Sentence Boundary Enforcement**

- Never cut mid-sentence
- Use ASR word-level timestamps
- Regex-based sentence detection (., ?, !)
- NLP sentence splitter (spaCy, NLTK)

**2. Semantic Completeness Check**

```python
def is_semantically_complete(clip_text):
    # Check for question-answer pairs
    if has_question(clip_text) and not has_answer(clip_text):
        return False

    # Check for incomplete thoughts
    # e.g., "Because..." without prior context
    connectives = ['because', 'therefore', 'however', 'so', 'but']
    first_word = clip_text.split()[0].lower()
    if first_word in connectives:
        return False

    # Check for dangling references
    # e.g., "This is important" without "this" being defined
    if has_unresolved_references(clip_text):
        return False

    return True
```

**3. Topic Coherence**

- Use BERTopic to extract main topic
- Ensure all sentences relate to main topic
- Reject clips spanning multiple unrelated topics

**4. Emotional Arc Closure**

- Detect emotional peaks (excitement, tension)
- Ensure clip doesn't end mid-peak
- Look for resolution or natural pause

### False Positive Reduction

**1. Blacklist Filters**

- No clips with excessive silence (>30% dead air)
- No clips with audio issues (glitches, distortion)
- No clips with inappropriate content (profanity filter)
- No clips from intro/outro segments (first/last 5%)

**2. Deduplication**

- Use perceptual hashing on audio
- Semantic similarity on transcripts
- Visual similarity on keyframes
- Reject clips >80% similar to existing

**3. Context Dependency Check**

```python
def requires_prior_context(clip, video):
    # Check if clip references earlier content
    clip_text = clip['transcript']

    # Pronoun resolution
    pronouns = extract_pronouns(clip_text)
    for pronoun in pronouns:
        referent = resolve_pronoun(pronoun, clip, video)
        if referent is None:  # Cannot resolve
            return True

    # Temporal references ("earlier", "previously")
    if has_temporal_back_reference(clip_text):
        return True

    # Requires setup (e.g., "The punchline is...")
    if requires_setup(clip_text, video):
        return True

    return False
```

**4. Minimum Quality Thresholds**

- Audio quality score > 0.7
- Visual quality score > 0.6
- Transcript confidence > 0.8
- Hook strength > 0.5

### Narrative Understanding Improvements

**1. Story Arc Detection**

```python
def detect_story_arc(video):
    transcript = video.transcript
    sentences = split_into_sentences(transcript)

    # Identify narrative stages
    setup = []
    buildup = []
    climax = []
    resolution = []

    # Use sentiment progression
    sentiments = [analyze_sentiment(s) for s in sentences]

    # Find climax (max emotional intensity)
    max_idx = np.argmax([s['intensity'] for s in sentiments])

    # Partition around climax
    setup = sentences[:max_idx//3]
    buildup = sentences[max_idx//3:max_idx]
    climax = sentences[max_idx:max_idx+10]
    resolution = sentences[max_idx+10:]

    return {
        'setup': setup,
        'buildup': buildup,
        'climax': climax,
        'resolution': resolution
    }

# Prefer clips that include climax
def score_narrative_fit(clip, story_arc):
    clip_text = clip['transcript']

    score = 0
    if overlaps(clip_text, story_arc['climax']):
        score += 2  # High priority
    if overlaps(clip_text, story_arc['buildup']):
        score += 1
    if overlaps(clip_text, story_arc['setup']):
        score += 0.5

    return score
```

**2. Setup-Payoff Pairing**

- Detect jokes: Setup (question/premise) → Punchline
- Ensure both are in clip
- Use pattern matching + ML classifier

**3. Multi-Segment Clips**

- Some stories require 2-3 non-consecutive segments
- Use clip stitching with transition detection
- Add text overlays like "Earlier..." or "Later..."

**4. Speaker Continuity**

- Identify main speaker(s)
- Ensure clip doesn't abruptly switch speakers
- Exception: Dialogue exchanges (back-and-forth)

---

## 6️⃣ MVP vs FULL SYSTEM

### MVP (Minimum Viable Product)

**Goal**: Proof of concept in 4-6 weeks

**Features:**

- ✅ Basic ASR (Whisper Medium)
- ✅ Audio energy detection (librosa)
- ✅ Simple scene change detection (OpenCV)
- ✅ Transcript chunking (fixed 60s windows)
- ✅ XGBoost scoring model
- ✅ Rule-based boundary refinement
- ✅ Top-K clip selection

**Limitations:**

- No global context modeling
- No domain adaptation
- No advanced narrative understanding
- Limited boundary optimization

**Tech Stack:**

- Python 3.10+
- Whisper (openai/whisper)
- librosa, OpenCV
- scikit-learn, XGBoost
- Flask API for inference

**Expected Performance:**

- 60-70% NDCG@10
- Works on clean audio podcasts
- 5-10 minutes processing time per hour of video
- CPU-only inference possible (slow)

---

### Production System (Full)

**Goal**: Production-grade system in 3-6 months

**Features:**

- ✅ Faster Whisper Large-v3 (optimized)
- ✅ Multi-modal embeddings (CLIP, CLAP)
- ✅ Hierarchical transformer for global context
- ✅ Domain-specific models (podcast, gaming, lecture)
- ✅ Neural ranker with contrastive learning
- ✅ Advanced boundary detection (NLP + emotion)
- ✅ Quality filters (coherence, hook strength)
- ✅ Multi-stage pipeline with caching
- ✅ Real-time processing capability
- ✅ A/B testing framework

**Enhancements:**

- Speaker diarization (who speaks when)
- Visual saliency detection (face focus, text overlays)
- Topic modeling with BERTopic
- Sentiment analysis timeline
- Hook generation (auto-edit first 3 seconds)
- Multi-language support
- Batch processing at scale
- Model monitoring & retraining pipeline

**Tech Stack:**

- Python 3.11+, PyTorch 2.0+
- Faster Whisper, HuggingFace Transformers
- CLIP, instructor embeddings
- XGBoost → Neural ranker
- Ray for distributed processing
- FastAPI for production API
- PostgreSQL for metadata
- Redis for caching
- MLflow for experiment tracking
- Docker + Kubernetes for deployment

**Expected Performance:**

- 80-85% NDCG@10
- Works across domains (podcast, gaming, vlog)
- 2-3 minutes processing time per hour of video (GPU)
- Handles 100+ videos/hour in batch mode

---

### Roadmap (What to Build When)

**Month 1-2: MVP**

- [ ] Basic feature extraction pipeline
- [ ] XGBoost scoring model
- [ ] Rule-based boundary detection
- [ ] Simple API endpoint
- [ ] Evaluation on 100 test videos

**Month 3: Context Modeling**

- [ ] Global video embeddings
- [ ] Topic segmentation
- [ ] Narrative structure analyzer
- [ ] Context-conditioned scoring

**Month 4: Quality & Boundaries**

- [ ] Semantic boundary refinement
- [ ] Coherence validation
- [ ] Hook detection
- [ ] False positive filters

**Month 5: Domain Adaptation**

- [ ] Video classifier
- [ ] Domain-specific models
- [ ] Multi-domain evaluation

**Month 6: Production Ready**

- [ ] Neural ranker
- [ ] Distributed processing
- [ ] A/B testing framework
- [ ] Model monitoring
- [ ] Auto-retraining pipeline

---

## 7️⃣ IMPLEMENTATION DETAILS

### File Structure

```
clipcatch-ai/
│
├── data/
│   ├── raw/                    # Raw videos
│   ├── processed/              # Extracted features
│   ├── annotations/            # Human labels
│   └── models/                 # Saved model weights
│
├── src/
│   ├── features/
│   │   ├── audio.py           # Audio feature extraction
│   │   ├── visual.py          # Visual feature extraction
│   │   ├── transcript.py      # ASR + text processing
│   │   └── multimodal.py      # Feature fusion
│   │
│   ├── models/
│   │   ├── context_encoder.py    # Global context model
│   │   ├── scorer.py              # Segment scoring
│   │   ├── ranker.py              # Final ranking
│   │   └── classifier.py          # Domain classifier
│   │
│   ├── pipeline/
│   │   ├── extraction.py      # Feature extraction pipeline
│   │   ├── scoring.py         # Scoring pipeline
│   │   ├── boundaries.py      # Boundary refinement
│   │   └── quality.py         # Quality filters
│   │
│   ├── utils/
│   │   ├── video_utils.py     # Video I/O
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualization.py   # Result visualization
│   │
│   └── api/
│       ├── app.py             # FastAPI server
│       └── schemas.py         # Request/response models
│
├── configs/
│   ├── mvp.yaml               # MVP configuration
│   └── production.yaml        # Production config
│
├── notebooks/
│   ├── exploration.ipynb      # Data exploration
│   └── experiments.ipynb      # Model experiments
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Batch inference
│
├── requirements.txt
├── Dockerfile
└── README.md
```

### Key Dependencies

```txt
# Core
python>=3.10
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# Audio
faster-whisper>=0.9.0
librosa>=0.10.0
pyAudioAnalysis>=0.3.14

# Video
opencv-python>=4.8.0
ffmpeg-python>=0.2.0

# ML
scikit-learn>=1.3.0
xgboost>=2.0.0
pytorch-lightning>=2.0.0

# NLP
spacy>=3.6.0
nltk>=3.8.0
bertopic>=0.15.0

# API
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Utils
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0

# Deployment
ray>=2.6.0
redis>=4.6.0
celery>=5.3.0
```

### Example API Usage

```python
# Client code
import requests

# Upload video
response = requests.post(
    "http://api.clipcatch.ai/analyze",
    json={
        "video_url": "https://youtube.com/watch?v=...",
        "num_clips": 10,
        "target_duration": 60,
        "domain": "podcast"  # auto-detect if not provided
    }
)

job_id = response.json()['job_id']

# Poll for results
while True:
    status = requests.get(f"http://api.clipcatch.ai/status/{job_id}")
    if status.json()['state'] == 'completed':
        break
    time.sleep(5)

# Get clips
clips = requests.get(f"http://api.clipcatch.ai/clips/{job_id}")
for clip in clips.json()['clips']:
    print(f"Clip: {clip['start']}s - {clip['end']}s")
    print(f"Score: {clip['score']}")
    print(f"Download: {clip['download_url']}")
```

---

## 8️⃣ COST & INFRASTRUCTURE

### Compute Requirements

**MVP (CPU-only):**

- Server: 16 CPU cores, 32GB RAM
- Processing: ~10 min per 1hr video
- Cost: ~$200/month (AWS c5.4xlarge)

**Production (GPU):**

- Server: 4x A100 GPUs, 128GB RAM
- Processing: ~2 min per 1hr video
- Cost: ~$8,000/month (AWS p4d.24xlarge)
- OR: Spot instances ~$3,000/month

**Batch Processing:**

- Use Ray on Kubernetes
- Autoscale based on queue depth
- Mix of GPU (Whisper) + CPU (scoring)

### Storage

**Per Video:**

- Raw video: ~500MB (1hr @ 720p)
- Audio: ~50MB
- Features: ~10MB
- Clips: ~200MB (10 clips)

**1000 videos/month:**

- Storage: ~750GB
- Cost: ~$20/month (S3 Standard)

---

## 9️⃣ FUTURE ENHANCEMENTS

**Short-term (3-6 months):**

- [ ] Auto-generate captions with styling
- [ ] Background music recommendation
- [ ] Thumbnail frame selection
- [ ] Multi-clip compilation (best-of reels)
- [ ] Real-time processing (live streams)

**Medium-term (6-12 months):**

- [ ] AI video editing (cuts, transitions, effects)
- [ ] Voice cloning for narration
- [ ] Face tracking & auto-cropping for vertical video
- [ ] Trend detection (what's going viral now)
- [ ] Personalized virality (per audience segment)

**Long-term (12+ months):**

- [ ] Generative AI for clip enhancement
  - Improve audio quality (denoise, enhance)
  - Upscale video resolution
  - Generate B-roll footage
- [ ] Multi-video remixing (mashups)
- [ ] Interactive clips (choose your own adventure)
- [ ] Live feedback loop (A/B test auto-optimization)

---

## 🔟 CONCLUSION

This system prioritizes **context-awareness** over naive segment scoring. By understanding the full video narrative, topic structure, and domain-specific patterns, we can extract clips that:

1. **Make sense standalone** (semantic completeness)
2. **Hook viewers immediately** (strong first 3 seconds)
3. **Fit the narrative** (setup-payoff, emotional arcs)
4. **Are domain-appropriate** (podcast ≠ gaming ≠ lecture)

The MVP can be built in 4-6 weeks with off-the-shelf models. The production system requires custom training but achieves 80%+ ranking accuracy and handles real-world scale.

**Key Differentiator**: Other tools cut random segments or use simple audio peaks. This system understands _why_ a clip is viral, _how_ it fits in context, and _what_ makes a good hook.

---

## 📚 REFERENCES

**Papers:**

- "Attention Is All You Need" (Transformers)
- "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
- "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision"
- "BERTopic: Neural Topic Modeling with a Class-based TF-IDF Procedure"
- "Learning to Rank for Information Retrieval" (XGBoost ranking)

**Libraries:**

- HuggingFace Transformers
- Faster Whisper (CTranslate2)
- Sentence Transformers
- librosa, OpenCV
- PyTorch Lightning

**Datasets:**

- YouTube-8M (video classification)
- VoxCeleb (speaker diarization)
- AudioSet (audio event detection)
- MELD (emotion recognition in dialogue)

---

**Author**: Context-Aware AI Engineering Team  
**Version**: 1.0  
**Last Updated**: December 23, 2025
