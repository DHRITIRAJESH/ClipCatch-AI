@echo off
echo ========================================
echo ClipCatch AI - Installation and Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found!
echo.

REM Check FFmpeg installation
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg is not installed or not in PATH
    echo Please install FFmpeg from https://ffmpeg.org/download.html
    echo The application will not work without FFmpeg!
    echo.
    pause
)

echo [2/4] FFmpeg found!
echo.

REM Install Python dependencies
echo [3/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [4/4] Starting backend server...
echo.
echo ========================================
echo Server will start on http://localhost:5000
echo Open index.html in your browser to use the app
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the Flask server
python app.py

pause
