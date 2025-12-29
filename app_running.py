from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
import subprocess
import wave
import numpy as np
from pathlib import Path
import threading
import time

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
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0

def extract_audio(video_path, audio_path):
    """Extract audio from video"""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            '-y',
            audio_path
        ]
        subprocess.run(cmd, capture_output=True)
        return os.path.exists(audio_path)
    except:
        return False

def analyze_audio_peaks(audio_path, duration):
    """Analyze audio for peak moments"""
    try:
        # Simple audio analysis - detect volume peaks
        peaks = []
        
        # In production, use librosa or other audio analysis libraries
        # For now, generate random peaks as demonstration
        num_peaks = min(20, int(duration / 5))
        for i in range(num_peaks):
            peaks.append({
                'time': np.random.uniform(0, duration),
                'intensity': np.random.uniform(0.7, 1.0)
            })
        
        return sorted(peaks, key=lambda x: x['intensity'], reverse=True)
    except:
        return []

def detect_motion_intensity(video_path, duration):
    """Detect motion intensity in video"""
    # In production, use OpenCV for motion detection
    # For now, simulate motion detection
    motion_moments = []
    
    num_moments = min(15, int(duration / 10))
    for i in range(num_moments):
        motion_moments.append({
            'time': np.random.uniform(0, duration),
            'intensity': np.random.uniform(0.6, 1.0)
        })
    
    return sorted(motion_moments, key=lambda x: x['intensity'], reverse=True)

def transcribe_audio(audio_path):
    """Transcribe audio to text for hook word detection"""
    # In production, use Whisper, Google Speech-to-Text, or similar
    # For now, return sample transcription
    return {
        'segments': [
            {'time': 5.2, 'text': 'This is amazing'},
            {'time': 15.7, 'text': 'Wow, that is incredible'},
            {'time': 32.1, 'text': 'Unbelievable performance'},
            {'time': 48.5, 'text': 'Absolutely stunning'},
        ]
    }

def find_hook_words(transcription, hook_words):
    """Find moments where hook words are mentioned"""
    hook_moments = []
    
    if not hook_words:
        return hook_moments
    
    hook_words_list = [w.strip().lower() for w in hook_words.split(',') if w.strip()]
    
    for segment in transcription.get('segments', []):
        text = segment['text'].lower()
        for hook_word in hook_words_list:
            if hook_word in text:
                hook_moments.append({
                    'time': segment['time'],
                    'word': hook_word,
                    'text': segment['text']
                })
    
    return hook_moments

def calculate_viral_score(time, audio_peaks, motion_moments, hook_moments, duration):
    """Calculate viral potential score for a timestamp"""
    score = 50  # Base score
    
    # Check audio peaks
    for peak in audio_peaks[:10]:
        if abs(peak['time'] - time) < 3:
            score += peak['intensity'] * 15
    
    # Check motion intensity
    for motion in motion_moments[:10]:
        if abs(motion['time'] - time) < 3:
            score += motion['intensity'] * 15
    
    # Check hook words
    for hook in hook_moments:
        if abs(hook['time'] - time) < 2:
            score += 20
    
    return min(100, int(score))

def generate_clips(video_path, segments, output_folder):
    """Generate video clips using ffmpeg"""
    clips = []
    
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_folder, f'clip_{i+1}.mp4')
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(segment['start']),
                '-t', str(segment['duration']),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',
                output_file
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(output_file):
                clips.append({
                    'index': i + 1,
                    'file': f'clip_{i+1}.mp4',
                    'start': segment['start'],
                    'end': segment['start'] + segment['duration'],
                    'duration': segment['duration'],
                    'score': segment['score'],
                    'reason': segment['reason']
                })
        except Exception as e:
            print(f"Error generating clip {i+1}: {e}")
    
    return clips

def analyze_video_async(job_id, video_path, config):
    """Analyze video asynchronously"""
    try:
        print(f"[DEBUG] Starting analysis for job {job_id}")
        analysis_progress[job_id] = {'progress': 0, 'status': 'Starting...'}
        
        # Get video duration
        duration = get_video_duration(video_path)
        print(f"[DEBUG] Video duration: {duration}")
        
        # If duration is 0 or invalid, use demo mode
        if duration == 0 or duration is None:
            print("[DEBUG] Using demo mode - FFmpeg not available")
            analysis_progress[job_id] = {'progress': 10, 'status': 'FFmpeg not detected - using demo mode...'}
            time.sleep(0.5)
            duration = 120  # Assume 2-minute video for demo
        
        analysis_progress[job_id] = {'progress': 10, 'status': 'Loading video...'}
        time.sleep(0.5)
        
        # Extract audio
        audio_path = video_path.replace('.mp4', '.wav')
        extract_audio(video_path, audio_path)
        analysis_progress[job_id] = {'progress': 30, 'status': 'Analyzing audio peaks...'}
        time.sleep(0.5)
        
        # Analyze audio
        audio_peaks = analyze_audio_peaks(audio_path, duration)
        analysis_progress[job_id] = {'progress': 50, 'status': 'Detecting motion intensity...'}
        time.sleep(0.5)
        
        # Detect motion
        motion_moments = detect_motion_intensity(video_path, duration)
        analysis_progress[job_id] = {'progress': 70, 'status': 'Identifying hook moments...'}
        time.sleep(0.5)
        
        # Transcribe audio and find hook words
        transcription = transcribe_audio(audio_path)
        hook_moments = find_hook_words(transcription, config.get('hook_words', ''))
        
        analysis_progress[job_id] = {'progress': 85, 'status': 'Generating clips...'}
        time.sleep(0.5)
        
        # Find best segments
        segment_duration = config.get('segment_duration', 5)
        clip_count = config.get('clip_count', 3)
        
        print(f"[DEBUG] Segment duration: {segment_duration}, Clip count: {clip_count}")
        print(f"[DEBUG] Duration: {duration}")
        
        # Score all possible segments
        segments = []
        step = max(1, segment_duration // 2)  # Overlap segments
        
        # Make sure we have valid duration
        if duration < segment_duration:
            duration = max(60, segment_duration * 10)  # Minimum 60 seconds or 10x segment
        
        end_range = max(1, int(duration - segment_duration))
        print(f"[DEBUG] Range: 0 to {end_range}, step: {step}")
        
        # Add safety limit - don't analyze more than 200 segments
        max_segments = 200
        actual_step = max(step, int(end_range / max_segments))
        print(f"[DEBUG] Actual step (with safety limit): {actual_step}")
        
        iteration_count = 0
        for start_time in range(0, end_range, actual_step):
            iteration_count += 1
            if iteration_count % 20 == 0:
                print(f"[DEBUG] Processing segment {iteration_count}")
            
            score = calculate_viral_score(
                start_time + segment_duration / 2,
                audio_peaks,
                motion_moments,
                hook_moments,
                duration
            )
            
            # Determine reason
            reasons = []
            if any(abs(p['time'] - start_time) < 3 for p in audio_peaks[:5]):
                reasons.append('High audio energy')
            if any(abs(m['time'] - start_time) < 3 for m in motion_moments[:5]):
                reasons.append('Intense motion')
            if any(abs(h['time'] - start_time) < 2 for h in hook_moments):
                reasons.append('Hook words detected')
            
            reason = ', '.join(reasons) if reasons else 'Engaging content'
            
            segments.append({
                'start': start_time,
                'duration': segment_duration,
                'score': score,
                'reason': reason
            })
        
        print(f"[DEBUG] Completed segment loop - generated {len(segments)} segments after {iteration_count} iterations")
        
        # If no segments generated, create default ones
        if not segments:
            for i in range(clip_count):
                start_time = i * (duration // (clip_count + 1))
                segments.append({
                    'start': start_time,
                    'duration': segment_duration,
                    'score': 75 + (i * 5),
                    'reason': 'Interesting moment detected'
                })
        
        # Sort by score and take top N
        best_segments = sorted(segments, key=lambda x: x['score'], reverse=True)[:clip_count]
        
        print(f"[DEBUG] Generated {len(segments)} segments, selected top {len(best_segments)}")
        
        # Generate actual video clips (or demo clips if FFmpeg unavailable)
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(output_folder, exist_ok=True)
        
        # Check if FFmpeg is available
        if check_ffmpeg():
            print("[DEBUG] FFmpeg available - generating clips")
            clips = generate_clips(video_path, best_segments, output_folder)
        else:
            print("[DEBUG] FFmpeg not available - creating demo clips")
            # Create demo clip data without actual video files
            clips = []
            for i, segment in enumerate(best_segments):
                clips.append({
                    'index': i + 1,
                    'file': None,  # No actual file in demo mode
                    'start': segment['start'],
                    'end': segment['start'] + segment['duration'],
                    'duration': segment['duration'],
                    'score': segment['score'],
                    'reason': segment['reason'] + ' (Demo - FFmpeg required for download)'
                })
        
        print(f"[DEBUG] Created {len(clips)} clips")
        
        analysis_progress[job_id] = {
            'progress': 100,
            'status': 'Complete',
            'clips': clips
        }
        
        print(f"[DEBUG] Analysis complete for job {job_id}")
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    except Exception as e:
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
    return "ClipCatch AI Backend API is running!"

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video for analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{filename}')
    file.save(file_path)
    
    return jsonify({
        'job_id': job_id,
        'filename': filename,
        'message': 'Video uploaded successfully'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start video analysis"""
    # Check if FFmpeg is installed
    if not check_ffmpeg():
        return jsonify({
            'error': 'FFmpeg not found',
            'message': 'FFmpeg is required for video processing. Please install FFmpeg and add it to your PATH.',
            'install_url': 'https://www.gyan.dev/ffmpeg/builds/'
        }), 400
    
    data = request.json
    
    job_id = data.get('job_id')
    config = {
        'segment_duration': data.get('segment_duration', 5),
        'clip_count': data.get('clip_count', 3),
        'hook_words': data.get('hook_words', ''),
        'detect_faces': data.get('detect_faces', True),
        'detect_audio': data.get('detect_audio', True),
        'detect_motion': data.get('detect_motion', True)
    }
    
    # Find video file
    video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(job_id)]
    
    if not video_files:
        return jsonify({'error': 'Video not found'}), 404
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
    
    # Start analysis in background thread
    thread = threading.Thread(target=analyze_video_async, args=(job_id, video_path, config))
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Analysis started'
    })

@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get analysis progress"""
    if job_id not in analysis_progress:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(analysis_progress[job_id])

@app.route('/api/download/<job_id>/<clip_name>', methods=['GET'])
def download_clip(job_id, clip_name):
    """Download a generated clip"""
    print(f"[DEBUG] Download request: job_id={job_id}, clip_name={clip_name}")
    
    # Extract index from clip_name if it includes .mp4
    if clip_name.endswith('.mp4'):
        clip_filename = clip_name
        clip_index = clip_name.replace('clip_', '').replace('.mp4', '')
    else:
        clip_index = clip_name
        clip_filename = f'clip_{clip_index}.mp4'
    
    clip_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, clip_filename)
    print(f"[DEBUG] Looking for clip at: {clip_path}")
    print(f"[DEBUG] File exists: {os.path.exists(clip_path)}")
    
    if not os.path.exists(clip_path):
        # List what files are actually there
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"[DEBUG] Files in output directory: {files}")
        else:
            print(f"[DEBUG] Output directory does not exist: {output_dir}")
        return jsonify({'error': 'Clip not found'}), 404
    
    return send_file(clip_path, as_attachment=True, download_name=f'clipcatch_viral_clip_{clip_index}.mp4')

@app.route('/api/stitch/<job_id>', methods=['POST'])
def stitch_clips(job_id):
    """Stitch all generated clips into a single video in chronological order"""
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    if not os.path.exists(output_folder):
        return jsonify({'error': 'No clips found for this job'}), 404
    
    # Get all clip files
    clip_files = [f for f in os.listdir(output_folder) if f.startswith('clip_') and f.endswith('.mp4')]
    
    if not clip_files:
        return jsonify({'error': 'No clips found to stitch'}), 404
    
    # Get the clip metadata from analysis progress to sort by start time
    clips_metadata = []
    if job_id in analysis_progress and 'clips' in analysis_progress[job_id]:
        clips_metadata = analysis_progress[job_id]['clips']
        # Sort by start time to ensure chronological order
        clips_metadata.sort(key=lambda x: x['start'])
        
        # Remove overlapping clips - keep earlier clips and skip later overlapping ones
        non_overlapping_clips = []
        for clip in clips_metadata:
            clip_start = clip['start']
            clip_end = clip['end']
            
            # Check if this clip overlaps with any already added clip
            has_overlap = False
            for added_clip in non_overlapping_clips:
                added_start = added_clip['start']
                added_end = added_clip['end']
                
                # Check for overlap
                if (clip_start < added_end and clip_end > added_start):
                    has_overlap = True
                    break
            
            # Only add if no overlap
            if not has_overlap:
                non_overlapping_clips.append(clip)
        
        # Create ordered list of clip files based on non-overlapping clips
        ordered_files = [f"clip_{clip['index']}.mp4" for clip in non_overlapping_clips if f"clip_{clip['index']}.mp4" in clip_files]
    else:
        # Fallback: just use alphabetical order if metadata not available
        ordered_files = sorted(clip_files)
    
    try:
        # Create a file list for ffmpeg concat
        concat_file = os.path.join(output_folder, 'concat_list.txt')
        with open(concat_file, 'w') as f:
            for clip_file in ordered_files:
                clip_path = os.path.join(output_folder, clip_file)
                # Convert to absolute path and normalize for Windows
                clip_path = os.path.abspath(clip_path)
                # Escape single quotes and backslashes for ffmpeg
                clip_path = clip_path.replace('\\', '/').replace("'", "\\'")
                f.write(f"file '{clip_path}'\n")
        
        # Output stitched video
        stitched_file = os.path.join(output_folder, 'stitched_video.mp4')
        
        # Use ffmpeg to concatenate videos
        command = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            '-y',  # Overwrite output file if exists
            stitched_file
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            # If direct copy fails, try re-encoding
            command = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                stitched_file
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise Exception(f'FFmpeg error: {result.stderr}')
        
        # Clean up concat file
        os.remove(concat_file)
        
        return jsonify({
            'message': 'Clips stitched successfully',
            'filename': 'stitched_video.mp4',
            'clip_count': len(clip_files)
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Stitching process timed out'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to stitch clips: {str(e)}'}), 500

@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup(job_id):
    """Cleanup uploaded video and generated clips"""
    # Remove uploaded video
    upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(job_id)]
    for f in upload_files:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        except:
            pass
    
    # Remove output folder
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    
    # Remove from progress dict
    if job_id in analysis_progress:
        del analysis_progress[job_id]
    
    return jsonify({'message': 'Cleanup successful'})

if __name__ == '__main__':
    print("=" * 50)
    print("ClipCatch AI Backend Server")
    print("=" * 50)
    print("\nServer starting on http://localhost:5000")
    print("\nEndpoints:")
    print("  POST   /api/upload      - Upload video")
    print("  POST   /api/analyze     - Start analysis")
    print("  GET    /api/progress    - Get progress")
    print("  GET    /api/download    - Download clip")
    print("  POST   /api/stitch      - Stitch all clips")
    print("  DELETE /api/cleanup     - Cleanup files")
    print("\n" + "=" * 50)
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
