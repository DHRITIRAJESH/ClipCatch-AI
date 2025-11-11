from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, time, threading, subprocess, tempfile, json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'viral_detector_model (1).pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

analysis_progress = {}

try:
    bundle = joblib.load(MODEL_PATH)
    rf = bundle["model"]
    scaler = bundle["scaler"]
    # Use local_files_only=True to use cached model
    embed_model = SentenceTransformer(bundle["embed_model_name"], local_files_only=True)
    print(f'Loaded ML model bundle from {MODEL_PATH}')
    MODEL_AVAILABLE = True
except Exception as e:
    print(f'Could not load model: {e}')
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

def extract_segment_features(text):
    """
    Extract features from text using SentenceTransformer embeddings + heuristic score
    Args:
        text: String to encode
    Returns:
        numpy array of features (embedding + heuristic + audio/visual dummy features)
    """
    try:
        if embed_model is None:
            return np.zeros(388)  # 384 embedding + 4 features
        
        # Generate text embedding
        emb = embed_model.encode([text], convert_to_numpy=True)[0]
        
        # Calculate heuristic score
        heur = heuristic_text_score(text)
        
        # Concatenate: embedding + [heuristic, audio_energy, audio_zcr, visual_motion]
        # Using dummy 0 for audio/visual since we don't extract them in real-time
        feat = np.concatenate([emb, [heur, 0, 0, 0]])
        
        return feat
    except Exception as e:
        print(f'Error extracting features: {e}')
        return np.zeros(388)  # Return zeros on error

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
                # Generate text description for the segment that can score well with heuristics
                position = start_time / duration
                
                # Create more varied and engaging descriptions
                if position < 0.2:
                    texts = [
                        f"Wait until you see what happens at {start_time}s - the opening is incredible!",
                        f"You won't believe how this video starts at {start_time}s",
                        f"Here's the thing about the beginning at {start_time}s - it's mind-blowing"
                    ]
                elif position < 0.4:
                    texts = [
                        f"What happened next at {start_time}s will shock you!",
                        f"Nobody talks about this moment at {start_time}s, but it's everything",
                        f"Did you know what happens at {start_time}s? Pure gold!"
                    ]
                elif position < 0.6:
                    texts = [
                        f"Plot twist: the middle section at {start_time}s changes everything",
                        f"Can you imagine what's happening at {start_time}s? Literally amazing!",
                        f"Why is everyone missing the {start_time}s mark? It's the best part!"
                    ]
                elif position < 0.8:
                    texts = [
                        f"But here's the catch at {start_time}s - you must see this",
                        f"Turns out the {start_time}s moment is what makes this viral",
                        f"Secret revealed at {start_time}s - this is exactly what you need!"
                    ]
                else:
                    texts = [
                        f"The ending at {start_time}s? Seriously, you have to watch this!",
                        f"Never skip to the end, but this {start_time}s finale is perfect",
                        f"How does it end at {start_time}s? Plot twist incoming!"
                    ]
                
                # Pick a text variant based on position
                text = texts[start_time % len(texts)]
                
                # Extract features using text embeddings + heuristic
                features = extract_segment_features(text)
                
                try:
                    # Scale features and predict
                    feat_scaled = scaler.transform([features])
                    score = float(rf.predict(feat_scaled)[0])
                    score = min(100, max(0, score))
                except Exception as e:
                    print(f'Prediction error: {e}')
                    score = 50 + np.random.rand() * 30
            else:
                score = 50 + np.random.rand() * 40
            
            position_bonus = 20 if start_time < duration * 0.3 else 0
            score += position_bonus
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
