# ClipCatch AI - Viral Clip Generator 🎬

A powerful AI-powered web application that analyzes videos and automatically generates viral-worthy clips.

## Features 🚀

- **Video Upload**: Drag & drop or browse to upload videos
- **AI Analysis**: Intelligent detection of:
  - Audio peaks and energy levels
  - Motion intensity in scenes
  - Hook words in speech (via transcription)
  - Viral potential scoring
- **Customizable Settings**:
  - Segment duration (5s, 10s, 15s, 30s, 60s)
  - Number of clips to generate (1-10)
  - Custom hook words
  - Advanced detection options
- **Real-time Progress**: Live progress tracking during analysis
- **Clip Preview**: Preview clips before downloading
- **Download**: Download individual clips as MP4 files

## Prerequisites 📋

### Required Software

1. **Python 3.8+**
2. **FFmpeg** (for video processing)
   - Windows: Download from https://ffmpeg.org/download.html
   - Add to PATH environment variable

### Install FFmpeg on Windows

1. Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH
4. Verify installation: `ffmpeg -version`

## Installation 🔧

1. **Install Python Dependencies**:

```bash
pip install -r requirements.txt
```

2. **For Production (Optional Advanced Features)**:

```bash
# For advanced audio analysis
pip install librosa

# For speech-to-text (requires PyTorch)
pip install openai-whisper torch

# For video/motion analysis
pip install opencv-python
```

## Usage 🎯

### 1. Start the Backend Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open the Frontend

Open `index.html` in your web browser or use:

```bash
start index.html
```

### 3. Generate Viral Clips

1. **Upload** your video file
2. **Configure** settings:
   - Choose segment duration
   - Set number of clips
   - Add custom hook words (optional)
3. **Click** "Analyze & Generate Clips"
4. **Wait** for AI analysis to complete
5. **Preview** and **Download** your viral clips!

## API Endpoints 🔌

### POST `/api/upload`

Upload a video file

- **Body**: FormData with 'video' file
- **Returns**: `{job_id, filename, message}`

### POST `/api/analyze`

Start video analysis

- **Body**:

```json
{
  "job_id": "string",
  "segment_duration": 5,
  "clip_count": 3,
  "hook_words": "amazing,wow,incredible",
  "detect_faces": true,
  "detect_audio": true,
  "detect_motion": true
}
```

### GET `/api/progress/<job_id>`

Get analysis progress

- **Returns**: `{progress, status, clips?}`

### GET `/api/download/<job_id>/<clip_index>`

Download a generated clip

- **Returns**: MP4 file

### DELETE `/api/cleanup/<job_id>`

Cleanup uploaded and generated files

## Project Structure 📁

```
ClipCatch AI/
├── index.html          # Frontend HTML
├── styles.css          # Styling
├── script.js           # Frontend JavaScript (with API integration)
├── app.py             # Flask backend server
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── uploads/           # Uploaded videos (auto-created)
└── outputs/           # Generated clips (auto-created)
```

## How It Works 🧠

1. **Upload**: Video is uploaded to the server
2. **Audio Extraction**: FFmpeg extracts audio track
3. **Audio Analysis**: Detects volume peaks and energy levels
4. **Motion Detection**: Analyzes video frames for motion intensity
5. **Transcription**: (Optional) Converts speech to text
6. **Hook Detection**: Finds moments with specified keywords
7. **Scoring**: Calculates viral potential for each segment
8. **Clip Generation**: FFmpeg creates top-scoring video clips
9. **Download**: User can download individual clips

## Configuration ⚙️

### Maximum File Size

Edit in `app.py`:

```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
```

### Supported Formats

```python
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
```

## Advanced Features (Optional) 🎓

### Add Speech-to-Text with Whisper

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe(audio_path)
```

### Add Motion Detection with OpenCV

```python
import cv2

# Implement frame difference analysis
# Detect scene changes and motion intensity
```

### Add Audio Analysis with Librosa

```python
import librosa

y, sr = librosa.load(audio_path)
# Analyze tempo, beats, spectral features
```

## Troubleshooting 🔍

### Backend not starting

- Make sure Python 3.8+ is installed
- Install all requirements: `pip install -r requirements.txt`
- Check if port 5000 is available

### FFmpeg not found

- Verify FFmpeg is installed: `ffmpeg -version`
- Add FFmpeg to System PATH
- Restart terminal/command prompt

### CORS errors

- Make sure Flask-CORS is installed
- Backend must be running on `http://localhost:5000`

### Upload fails

- Check file size (max 500MB by default)
- Verify file format is supported
- Check backend server logs

## Performance Tips 💡

- For faster processing, use shorter videos
- Reduce segment duration for more precise clips
- Disable unused detection options
- Use SSD storage for uploads/outputs folders

## Future Enhancements 🚀

- [ ] Real AI model integration (Whisper, YOLO, etc.)
- [ ] Cloud storage support (AWS S3, Azure)
- [ ] Batch processing multiple videos
- [ ] Social media optimization presets
- [ ] Automatic subtitle generation
- [ ] Face recognition and tracking
- [ ] Scene change detection
- [ ] Background music addition
- [ ] Video effects and filters

## License 📄

This project is open source and available for educational purposes.

## Support 💬

For issues or questions, please check:

1. This README
2. Backend server logs
3. Browser console errors

---

**Made with ❤️ for content creators**
