"""
Flask app with Context-Aware Viral Clip Detection
Integrates the new AI model
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
import subprocess
import threading
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from src.viral_clip_extractor import ViralClipExtractor

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Store analysis progress
analysis_progress = {}

# Initialize the AI model
viral_extractor = ViralClipExtractor(window_size=0.5, sample_rate=1.0)


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_duration(video_path):
    """Get video duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            return 0
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0


def generate_clip_file(video_path, start, duration, output_file):
    """Generate a single clip file using ffmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return os.path.exists(output_file)
    except Exception as e:
        print(f"Error generating clip: {e}")
        return False


def progress_callback(job_id, stage, progress):
    """Callback for AI model progress updates"""
    # Map progress to percentage
    stage_map = {
        "Extracting audio features": (0.0, 0.2),
        "Extracting visual features": (0.2, 0.4),
        "Extracting transcript": (0.4, 0.6),
        "Analyzing video context": (0.6, 0.65),
        "Generating candidate segments": (0.65, 0.68),
        "Scoring segments": (0.68, 0.75),
        "Refining clip boundaries": (0.75, 0.85),
        "Filtering and ranking clips": (0.85, 0.95),
        "Complete": (1.0, 1.0)
    }
    
    if stage in stage_map:
        start_pct, end_pct = stage_map[stage]
        actual_progress = start_pct + (end_pct - start_pct) * progress
        percentage = int(actual_progress * 100)
    else:
        percentage = int(progress * 100)
    
    if job_id in analysis_progress:
        analysis_progress[job_id]['progress'] = percentage
        analysis_progress[job_id]['status'] = stage


def analyze_video_async(job_id, video_path, config):
    """Analyze video using context-aware AI model"""
    try:
        print(f"[AI] Starting context-aware analysis for job {job_id}")
        analysis_progress[job_id] = {'progress': 0, 'status': 'Initializing AI model...'}
        
        # Get video metadata if available
        video_metadata = {
            'title': config.get('title', ''),
            'description': config.get('description', '')
        }
        
        # Extract clips using AI model
        num_clips = config.get('clip_count', 10)
        target_duration = config.get('segment_duration', 60)
        
        def update_progress(stage, progress):
            progress_callback(job_id, stage, progress)
        
        result = viral_extractor.extract_clips(
            video_path=video_path,
            num_clips=num_clips,
            target_duration=target_duration,
            video_metadata=video_metadata,
            progress_callback=update_progress
        )
        
        print(f"[AI] Extraction complete. Found {len(result['clips'])} clips")
        print(f"[AI] Domain detected: {result['domain']}")
        print(f"[AI] Topics: {result['topics']}")
        
        # Generate clip files
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(output_folder, exist_ok=True)
        
        clips_data = []
        
        if check_ffmpeg():
            print("[AI] Generating clip files with FFmpeg...")
            for i, clip in enumerate(result['clips']):
                output_file = os.path.join(output_folder, f'clip_{i+1}.mp4')
                
                success = generate_clip_file(
                    video_path,
                    clip['start'],
                    clip['duration'],
                    output_file
                )
                
                clips_data.append({
                    'index': i + 1,
                    'file': f'clip_{i+1}.mp4' if success else None,
                    'start': clip['start'],
                    'end': clip['end'],
                    'duration': clip['duration'],
                    'score': int(clip['score'] * 100),  # Convert to 0-100
                    'reason': clip['reason'],
                    'text': clip.get('text', '')[:200],  # First 200 chars
                    'hook_score': clip.get('hook_score', 0),
                    'domain': clip.get('domain', result['domain'])
                })
        else:
            print("[AI] FFmpeg not available - demo mode")
            for i, clip in enumerate(result['clips']):
                clips_data.append({
                    'index': i + 1,
                    'file': None,
                    'start': clip['start'],
                    'end': clip['end'],
                    'duration': clip['duration'],
                    'score': int(clip['score'] * 100),
                    'reason': clip['reason'] + ' (FFmpeg required for download)',
                    'text': clip.get('text', '')[:200],
                    'hook_score': clip.get('hook_score', 0),
                    'domain': clip.get('domain', result['domain'])
                })
        
        analysis_progress[job_id] = {
            'progress': 100,
            'status': 'Complete',
            'clips': clips_data,
            'domain': result['domain'],
            'topics': result['topics'],
            'stats': result['stats'],
            'processing_time': result['processing_time']
        }
        
        print(f"[AI] Analysis complete for job {job_id}")
        print(f"[AI] Processing time: {result['processing_time']:.1f}s")
        
    except Exception as e:
        import traceback
        print(f"[AI] Error in analysis: {e}")
        print(traceback.format_exc())
        
        analysis_progress[job_id] = {
            'progress': 0,
            'status': 'Error',
            'error': str(e)
        }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/api/')
def api_index():
    return jsonify({
        'name': 'ClipCatch AI - Context-Aware Edition',
        'version': '2.0',
        'model': 'Context-Aware Viral Detection',
        'ffmpeg': check_ffmpeg()
    })


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())
        filename = secure_filename(f'{job_id}_{file.filename}')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis"""
    data = request.json
    job_id = data.get('job_id')
    filepath = data.get('filepath')
    
    if not job_id or not filepath:
        return jsonify({'error': 'Missing job_id or filepath'}), 400
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Get configuration
    config = {
        'segment_duration': data.get('segment_duration', 60),
        'clip_count': data.get('clip_count', 10),
        'title': data.get('title', ''),
        'description': data.get('description', '')
    }
    
    # Start analysis in background
    thread = threading.Thread(
        target=analyze_video_async,
        args=(job_id, filepath, config)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'Analysis started'
    })


@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get analysis progress"""
    if job_id in analysis_progress:
        return jsonify(analysis_progress[job_id])
    else:
        return jsonify({'error': 'Job not found'}), 404


@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_clip(job_id, filename):
    """Download a generated clip"""
    clip_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename)
    
    if os.path.exists(clip_path):
        return send_file(clip_path, as_attachment=True)
    else:
        return jsonify({'error': 'Clip not found'}), 404


@app.route('/api/stitch/<job_id>', methods=['POST'])
def stitch_clips(job_id):
    """Stitch all clips into one video"""
    if job_id not in analysis_progress:
        return jsonify({'error': 'Job not found'}), 404
    
    if 'clips' not in analysis_progress[job_id]:
        return jsonify({'error': 'No clips available'}), 400
    
    clips_metadata = analysis_progress[job_id]['clips']
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    # Filter clips that have files
    available_clips = [c for c in clips_metadata if c.get('file')]
    
    if not available_clips:
        return jsonify({'error': 'No clip files available'}), 400
    
    # Sort chronologically by start time
    available_clips.sort(key=lambda x: x['start'])
    
    # Remove overlapping clips (keep earlier ones)
    non_overlapping = []
    for clip in available_clips:
        clip_start = clip['start']
        clip_end = clip['end']
        
        overlaps = False
        for added_clip in non_overlapping:
            added_start = added_clip['start']
            added_end = added_clip['end']
            
            if (clip_start < added_end and clip_end > added_start):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(clip)
    
    # Create concat file
    concat_file = os.path.join(output_folder, 'concat_list.txt')
    with open(concat_file, 'w') as f:
        for clip in non_overlapping:
            clip_path = os.path.join(output_folder, clip['file'])
            # Use forward slashes for ffmpeg
            clip_path = clip_path.replace('\\', '/')
            f.write(f"file '{clip_path}'\n")
    
    # Stitch using ffmpeg
    stitched_file = os.path.join(output_folder, 'stitched_video.mp4')
    
    try:
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            '-y',
            stitched_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        
        if os.path.exists(stitched_file):
            return send_file(stitched_file, as_attachment=True, 
                           download_name='stitched_video.mp4')
        else:
            return jsonify({'error': 'Failed to create stitched video'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Stitching failed: {str(e)}'}), 500


@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job files"""
    # Remove from progress
    if job_id in analysis_progress:
        del analysis_progress[job_id]
    
    # Remove upload file
    upload_folder = app.config['UPLOAD_FOLDER']
    for file in os.listdir(upload_folder):
        if file.startswith(job_id):
            try:
                os.remove(os.path.join(upload_folder, file))
            except:
                pass
    
    # Remove output folder
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if os.path.exists(output_folder):
        import shutil
        try:
            shutil.rmtree(output_folder)
        except:
            pass
    
    return jsonify({'status': 'Cleaned up'})


if __name__ == '__main__':
    print("=" * 60)
    print("ClipCatch AI - Context-Aware Viral Clip Detection")
    print("=" * 60)
    print(f"FFmpeg available: {check_ffmpeg()}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("Model: Context-Aware Multi-Modal Analysis")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
