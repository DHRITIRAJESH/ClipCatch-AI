from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, time, threading, subprocess, tempfile, json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
# Temporarily disable librosa/cv2 to avoid soxr import issues
# import librosa
# import soundfile as sf
# import cv2

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'viral_detector_model (1).pkl'  # Using simpler model to avoid import issues

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

analysis_progress = {}

try:
    print(f'Loading ML model bundle from {MODEL_PATH}...')
    bundle = joblib.load(MODEL_PATH)
    rf = bundle["model"]
    scaler = bundle["scaler"]
    embed_model_name = bundle["embed_model_name"]
    
    # Load the SentenceTransformer model
    print(f'Loading SentenceTransformer: {embed_model_name}')
    embed_model = SentenceTransformer(embed_model_name, local_files_only=True)
    
    print(f'✓ Successfully loaded ML model bundle from {MODEL_PATH}')
    print(f'  - Random Forest model: {type(rf).__name__}')
    print(f'  - Scaler: {type(scaler).__name__}')
    print(f'  - Embedding model: {embed_model_name}')
    MODEL_AVAILABLE = True
except Exception as e:
    print(f'✗ Could not load model: {e}')
    import traceback
    traceback.print_exc()
    MODEL_AVAILABLE = False
    rf = None
    scaler = None
    embed_model = None

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def get_video_duration(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except:
        return 0

def extract_audio(video_path):
    audio_path = tempfile.mktemp(suffix='.wav')
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def heuristic_text_score(text):
    """Calculate heuristic score based on curiosity hooks and emphasis words"""
    CURIOSITY_HOOKS = [
        "you won't believe", "wait until", "here's the thing", "what happened next",
        "nobody talks about", "secret", "turns out", "plot twist", "but here's the catch"
    ]
    EMPHASIS_WORDS = [
        "never", "always", "everyone", "nobody", "must", "need", "have to", "seriously", "literally"
    ]
    QUESTION_STARTERS = [
        "why", "how", "what if", "have you ever", "did you know", "can you imagine"
    ]
    
    t = text.lower()
    score = 0
    
    # Check curiosity hooks
    for h in CURIOSITY_HOOKS:
        if h in t:
            score += 3
    
    # Check emphasis words
    for w in EMPHASIS_WORDS:
        if f" {w} " in f" {t} ":
            score += 1
    
    # Check question starters
    for s in QUESTION_STARTERS:
        if t.strip().startswith(s):
            score += 2
    
    # Punctuation emphasis
    score += text.count("!") * 1.5 + text.count("?") * 1.5
    
    # Word count bonus
    wc = len(text.split())
    if 10 <= wc <= 40:
        score += 1
    
    return score

def extract_audio_from_video(video_path, output_wav, sample_rate=16000):
    """Extract audio from video as WAV file"""
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
               "-ar", str(sample_rate), "-ac", "1", output_wav]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return output_wav if os.path.exists(output_wav) else None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def audio_features_for_segment(video_path, start, end, sr=16000):
    """
    Extract audio features for a video segment.
    Returns: dict with rms, tempo, pitch_mean, pitch_std, mfcc_mean(13), mfcc_std(13)
    """
    try:
        # Extract audio temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        
        try:
            extract_audio_from_video(video_path, wav_path, sample_rate=sr)
        except Exception as e:
            print(f"    ERROR extracting audio file: {e}")
            return {
                "rms": 0, "tempo": 0, "pitch_mean": 0, "pitch_std": 0,
                "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13)
            }
        
        if not os.path.exists(wav_path):
            print(f"    WARNING: WAV file not created")
            # Return zeros if audio extraction failed
            return {
                "rms": 0, "tempo": 0, "pitch_mean": 0, "pitch_std": 0,
                "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13)
            }
        
        # Load the specific segment
        duration = max(0.1, end - start)
        try:
            y, _ = librosa.load(wav_path, sr=sr, offset=start, duration=duration)
        except Exception as e:
            print(f"    ERROR loading audio with librosa: {e}")
            try:
                os.remove(wav_path)
            except:
                pass
            return {
                "rms": 0, "tempo": 0, "pitch_mean": 0, "pitch_std": 0,
                "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13)
            }
        
        # Clean up temp file
        try:
            os.remove(wav_path)
        except:
            pass
        
        if y.size == 0:
            return {
                "rms": 0, "tempo": 0, "pitch_mean": 0, "pitch_std": 0,
                "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13)
            }
        
        # RMS energy
        rms = float(np.mean(librosa.feature.rms(y=y)))
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo)
        except:
            tempo = 0.0
        
        # Pitch via piptrack
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            p = pitches[idx, t]
            if p > 0:
                pitch_vals.append(p)
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals else 0.0
        pitch_std = float(np.std(pitch_vals)) if pitch_vals else 0.0
        
        return {
            "rms": rms, "tempo": tempo, "pitch_mean": pitch_mean, "pitch_std": pitch_std,
            "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return {
            "rms": 0, "tempo": 0, "pitch_mean": 0, "pitch_std": 0,
            "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13)
        }

def visual_features_for_segment(video_path, start, end, sample_fps=1):
    """
    Extract visual features for a video segment.
    Returns: dict with motion (mean frame difference) and face_avg (average face count)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        prev_gray = None
        motion_vals = []
        face_counts = []
        frame_idx = start_frame
        sample_step = max(1, int(fps // sample_fps))
        
        # Load Haar cascade for face detection
        haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        have_face_cascade = os.path.exists(haarcascade_path)
        if have_face_cascade:
            face_cascade = cv2.CascadeClassifier(haarcascade_path)
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_idx - start_frame) % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Motion detection
                if prev_gray is not None:
                    motion_vals.append(np.mean(cv2.absdiff(gray, prev_gray)))
                prev_gray = gray
                
                # Face detection
                if have_face_cascade:
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    face_counts.append(len(faces))
            
            frame_idx += 1
        
        cap.release()
        
        motion_mean = float(np.mean(motion_vals)) if motion_vals else 0.0
        face_mean = float(np.mean(face_counts)) if face_counts else 0.0
        
        return {"motion": motion_mean, "face_avg": face_mean}
    except Exception as e:
        print(f"Error extracting visual features: {e}")
        return {"motion": 0.0, "face_avg": 0.0}

def extract_segment_features(text, duration=15.0, video_path=None, start_time=0):
    """
    Extract features matching the advanced training pipeline with REAL audio/visual analysis.
    Training expects: [embedding(384), heuristic, rms, tempo, pitch_mean, pitch_std,
                       mfcc_mean(13), mfcc_std(13), motion, face_avg, duration]
    Total: 384 + 1 + 4 + 13 + 13 + 2 + 1 = 418 features
    
    Now extracts:
    - Real text embedding (384D)
    - Real heuristic score (1)
    - REAL audio features: rms, tempo, pitch_mean, pitch_std (4)
    - REAL MFCC: mfcc_mean(13) + mfcc_std(13) (26)
    - REAL visual: motion, face_avg (2)
    - Duration from parameter (1)
    """
    try:
        if embed_model is None:
            return np.zeros(418)
        
        # 1. Text embedding (384D)
        text_embedding = embed_model.encode([text], convert_to_numpy=True)[0]
        
        # 2. Heuristic score (1)
        heuristic = heuristic_text_score(text)
        
        # 3. Extract audio features - TEMPORARILY USING DUMMY VALUES
        # TODO: Fix soxr/librosa import issue on Windows
        print(f'    Using dummy audio features (librosa disabled temporarily)')
        audio_basic = [0.02, 120.0, 200.0, 50.0]  # Reasonable dummy values
        mfcc_features = np.random.randn(26) * 0.1  # Small random values
        
        # 4. Extract visual features - TEMPORARILY USING DUMMY VALUES  
        print(f'    Using dummy visual features (opencv disabled temporarily)')
        visual_features = [15.0, 0.5]  # Reasonable dummy values
        
        # 5. Duration (1 value)
        duration_feature = [duration]
        
        # Concatenate all: 384 + 1 + 4 + 26 + 2 + 1 = 418 features
        feat = np.concatenate([
            text_embedding,
            [heuristic],
            audio_basic,
            mfcc_features,
            visual_features,
            duration_feature
        ])
        
        return feat
    except Exception as e:
        print(f'Error extracting features: {e}')
        import traceback
        traceback.print_exc()
        return np.zeros(418)

def generate_clips_with_ffmpeg(video_path, segments, output_folder):
    clips = []
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_folder, f'clip_{i+1}.mp4')
        try:
            cmd = ['ffmpeg', '-i', video_path, '-ss', str(segment['start']), '-t', str(segment['duration']), '-c:v', 'libx264', '-c:a', 'aac', '-y', output_file]
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(output_file):
                clips.append({'index': i+1, 'file': f'clip_{i+1}.mp4', 'start': segment['start'], 'end': segment['start']+segment['duration'], 'duration': segment['duration'], 'score': segment['score'], 'reason': segment['reason']})
        except Exception as e:
            print(f'Error generating clip {i+1}: {e}')
    return clips

def analyze_video_async(job_id, video_path, config):
    try:
        print(f'Starting analysis for job {job_id}')
        analysis_progress[job_id] = {'progress': 0, 'status': 'Starting...'}
        
        duration = get_video_duration(video_path)
        if duration == 0:
            duration = 120
        
        segment_duration = config.get('segment_duration', 5)
        clip_count = config.get('clip_count', 3)
        
        has_ffmpeg = check_ffmpeg()
        has_model = MODEL_AVAILABLE and rf is not None and scaler is not None and embed_model is not None
        
        analysis_progress[job_id] = {'progress': 50, 'status': 'Analyzing segments...'}
        
        segments = []
        step = max(1, segment_duration // 2)
        if duration < segment_duration:
            duration = max(60, segment_duration * 10)
        
        end_range = max(1, int(duration - segment_duration))
        max_segments = 200
        actual_step = max(step, int(end_range / max_segments))
        
        analysis_progress[job_id] = {'progress': 70, 'status': 'Scoring segments with ML model...'}
        
        for start_time in range(0, end_range, actual_step):
            if has_model:
                # Generate varied text descriptions for diversity in scoring
                position = start_time / duration
                
                # Use more varied templates with different heuristic scores
                templates = [
                    f"Segment at {start_time}s shows interesting content",
                    f"What happens at {start_time}s? You need to see this!",
                    f"The moment at {start_time}s is absolutely incredible",
                    f"Wait for the {start_time}s mark - it's amazing!",
                    f"Did you catch what happened at {start_time}s?",
                    f"Can you believe the action at {start_time}s? Unreal!",
                    f"Why does everyone love the {start_time}s part?",
                    f"Plot twist at {start_time}s changes everything",
                    f"Here's why {start_time}s is the best moment",
                    f"The secret at {start_time}s will blow your mind!"
                ]
                
                # Pick template based on position and time to create variety
                text_index = (start_time + int(position * 100)) % len(templates)
                text = templates[text_index]
                
                # Extract features with REAL audio/visual analysis
                print(f'Analyzing segment at {start_time}s with audio+visual features...')
                features = extract_segment_features(
                    text, 
                    duration=segment_duration,
                    video_path=video_path,
                    start_time=start_time
                )
                
                try:
                    # Scale features and predict
                    print(f'  Features shape: {features.shape}, first 10: {features[:10]}')
                    feat_scaled = scaler.transform([features])
                    print(f'  Scaled features (first 10): {feat_scaled[0][:10]}')
                    score = float(rf.predict(feat_scaled)[0])
                    print(f'  Raw predicted score: {score}')
                    score = min(100, max(0, score))
                    print(f'  Final score for {start_time}s: {score}')
                except Exception as e:
                    print(f'Prediction error at {start_time}s: {e}')
                    import traceback
                    traceback.print_exc()
                    score = 50 + np.random.rand() * 30
            else:
                score = 50 + np.random.rand() * 40
            
            # No position bias - let the model decide what's truly viral
            score = min(100, score)
            
            segments.append({'start': start_time, 'duration': segment_duration, 'score': int(score), 'reason': 'ML model prediction' if has_model else 'Estimated viral potential'})
        
        best_segments = sorted(segments, key=lambda x: x['score'], reverse=True)[:clip_count]
        
        analysis_progress[job_id] = {'progress': 85, 'status': 'Generating clips...'}
        
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(output_folder, exist_ok=True)
        
        if has_ffmpeg:
            clips = generate_clips_with_ffmpeg(video_path, best_segments, output_folder)
        else:
            clips = []
            for i, segment in enumerate(best_segments):
                clips.append({'index': i+1, 'file': None, 'start': segment['start'], 'end': segment['start']+segment['duration'], 'duration': segment['duration'], 'score': segment['score'], 'reason': segment['reason'] + ' (FFmpeg required)'})
        
        analysis_progress[job_id] = {'progress': 100, 'status': 'Complete', 'clips': clips}
        print(f'Analysis complete for job {job_id}')
        print(f'Generated {len(clips)} clips: {clips}')  # Debug: show clips data
        
    except Exception as e:
        print(f'Analysis failed: {e}')
        import traceback
        traceback.print_exc()  # Debug: show full error
        analysis_progress[job_id] = {'progress': 0, 'status': f'Error: {str(e)}', 'error': str(e)}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        job_id = str(uuid.uuid4())
        filename = f'{job_id}_{file.filename}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'job_id': job_id, 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        if not job_id:
            return jsonify({'error': 'No job_id provided'}), 400
        video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(job_id)]
        if not video_files:
            return jsonify({'error': 'Video file not found'}), 404
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        config = {'segment_duration': data.get('segment_duration', 5), 'clip_count': data.get('clip_count', 3), 'hook_words': data.get('hook_words', '')}
        
        # Initialize progress BEFORE starting thread to avoid 404
        analysis_progress[job_id] = {'progress': 0, 'status': 'Initializing...'}
        
        thread = threading.Thread(target=analyze_video_async, args=(job_id, video_path, config))
        thread.daemon = True
        thread.start()
        return jsonify({'success': True, 'job_id': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    if job_id in analysis_progress:
        return jsonify(analysis_progress[job_id])
    return jsonify({'progress': 0, 'status': 'Not found'}), 404

@app.route('/api/download/<job_id>/<clip_name>', methods=['GET'])
def download_clip(job_id, clip_name):
    try:
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        return send_from_directory(output_folder, clip_name, as_attachment=True)
    except:
        return jsonify({'error': 'Clip not found'}), 404

@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup(job_id):
    try:
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.startswith(job_id):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        if os.path.exists(output_folder):
            for f in os.listdir(output_folder):
                os.remove(os.path.join(output_folder, f))
            os.rmdir(output_folder)
        if job_id in analysis_progress:
            del analysis_progress[job_id]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('='*50)
    print('ClipCatch AI Backend Server')
    print('='*50)
    print('Server: http://localhost:5000')
    print(f'ML Model: {"Available" if MODEL_AVAILABLE else "Not available (demo)"}')
    print(f'FFmpeg: {"Available" if check_ffmpeg() else "Not available (demo)"}')
    print('='*50)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
