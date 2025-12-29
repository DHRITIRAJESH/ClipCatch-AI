// Global variables
let uploadedVideo = null;
let selectedDuration = 5;
let clipCount = 3;
let currentJobId = null;

// API Configuration
const API_URL = 'http://localhost:5000/api';

// DOM Elements
const videoInput = document.getElementById('videoInput');
const uploadArea = document.getElementById('uploadArea');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const videoPreview = document.getElementById('videoPreview');
const previewVideo = document.getElementById('previewVideo');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const clipsGrid = document.getElementById('clipsGrid');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const clipCountSlider = document.getElementById('clipCount');
const clipCountValue = document.getElementById('clipCountValue');
const durationBtns = document.querySelectorAll('.duration-btn');
const stitchBtn = document.getElementById('stitchBtn');

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(99, 102, 241, 0.1)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('video/')) {
        handleVideoUpload(files[0]);
    }
});

// File input change
videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleVideoUpload(e.target.files[0]);
    }
});

// Handle video upload
async function handleVideoUpload(file) {
    uploadedVideo = file;
    fileName.textContent = file.name;
    fileInfo.style.display = 'flex';
    
    // Show video preview
    const videoURL = URL.createObjectURL(file);
    previewVideo.src = videoURL;
    videoPreview.style.display = 'block';
    
    // Hide upload area text
    uploadArea.querySelector('h2').style.display = 'none';
    uploadArea.querySelector('p').style.display = 'none';
    uploadArea.querySelector('.btn-browse').style.display = 'none';
    uploadArea.querySelector('.upload-icon').style.display = 'none';
    
    // Upload to server
    showNotification('Uploading video to server...', 'info');
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        currentJobId = data.job_id;
        analyzeBtn.disabled = false;
        
        showNotification('Video uploaded successfully!', 'success');
    } catch (error) {
        showNotification('Upload failed. Make sure the backend server is running!', 'error');
        console.error('Upload error:', error);
        analyzeBtn.disabled = true;
    }
}

// Remove file
removeFile.addEventListener('click', async (e) => {
    e.stopPropagation();
    
    // Cleanup server files
    if (currentJobId) {
        try {
            await fetch(`${API_URL}/cleanup/${currentJobId}`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }
    
    uploadedVideo = null;
    currentJobId = null;
    videoInput.value = '';
    fileInfo.style.display = 'none';
    videoPreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    
    // Show upload area text
    uploadArea.querySelector('h2').style.display = 'block';
    uploadArea.querySelector('p').style.display = 'block';
    uploadArea.querySelector('.btn-browse').style.display = 'inline-flex';
    uploadArea.querySelector('.upload-icon').style.display = 'block';
});

// Duration button selection
durationBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        durationBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedDuration = parseInt(btn.dataset.duration);
        document.getElementById('segmentDuration').value = selectedDuration;
    });
});

// Clip count slider
clipCountSlider.addEventListener('input', (e) => {
    clipCount = parseInt(e.target.value);
    clipCountValue.textContent = clipCount;
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedVideo || !currentJobId) {
        showNotification('Please upload a video first!', 'error');
        return;
    }
    
    // Get configuration
    const hookWords = document.getElementById('hookWords').value;
    const detectFaces = document.getElementById('detectFaces').checked;
    const detectAudio = document.getElementById('detectAudio').checked;
    const detectMotion = document.getElementById('detectMotion').checked;
    
    // Show loading
    analyzeBtn.style.display = 'none';
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    
    try {
        // Start analysis
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_id: currentJobId,
                segment_duration: selectedDuration,
                clip_count: clipCount,
                hook_words: hookWords,
                detect_faces: detectFaces,
                detect_audio: detectAudio,
                detect_motion: detectMotion
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Handle FFmpeg not installed error
            if (data.error === 'FFmpeg not found') {
                showNotification(
                    `❌ FFmpeg is required!\n\n${data.message}\n\nDownload from: ${data.install_url}`, 
                    'error'
                );
                loading.style.display = 'none';
                analyzeBtn.style.display = 'inline-flex';
                return;
            }
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Poll for progress
        await pollProgress();
        
    } catch (error) {
        showNotification('Analysis failed: ' + error.message, 'error');
        console.error('Analysis error:', error);
        loading.style.display = 'none';
        analyzeBtn.style.display = 'inline-flex';
    }
});

// Poll for analysis progress
async function pollProgress() {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/progress/${currentJobId}`);
            if (!response.ok) throw new Error('Failed to get progress');
            
            const data = await response.json();
            
            // Update progress bar
            progressFill.style.width = data.progress + '%';
            progressText.textContent = data.progress + '%';
            loading.querySelector('p').textContent = data.status;
            
            // Check if complete
            if (data.progress === 100 && data.status === 'Complete') {
                console.log('Analysis complete! Received data:', data);
                console.log('Clips array:', data.clips);
                console.log('Type of clips:', typeof data.clips);
                console.log('First clip:', data.clips && data.clips[0]);
                console.log('ALL clips stringified:', data.clips && JSON.stringify(data.clips, null, 2));
                clearInterval(pollInterval);
                displayResults(data.clips);
                loading.style.display = 'none';
                analyzeBtn.style.display = 'inline-flex';
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            // Check for errors
            if (data.status === 'Error') {
                clearInterval(pollInterval);
                showNotification('Analysis error: ' + data.error, 'error');
                loading.style.display = 'none';
                analyzeBtn.style.display = 'inline-flex';
            }
            
        } catch (error) {
            clearInterval(pollInterval);
            showNotification('Failed to get progress', 'error');
            loading.style.display = 'none';
            analyzeBtn.style.display = 'inline-flex';
        }
    }, 1000); // Poll every second
}

// Display results
function displayResults(clips) {
    // FORCE complete clear of old content
    clipsGrid.innerHTML = '';
    clipsGrid.textContent = ''; // Extra clear
    
    // Debug: Log what we received
    console.log('displayResults called with:', clips);
    console.log('Is array?', Array.isArray(clips));
    console.log('Length:', clips ? clips.length : 'clips is null/undefined');
    
    // Validate clips
    if (!clips || !Array.isArray(clips) || clips.length === 0) {
        clipsGrid.innerHTML = '<p style="color: #999; text-align: center; padding: 2rem;">No clips generated. Try adjusting your settings.</p>';
        return;
    }
    
    clips.forEach((clip, index) => {
        console.log(`Processing clip ${index}:`, clip);
        const clipCard = createClipCard(clip);
        clipsGrid.appendChild(clipCard);
    });
    
    // Show stitch button
    if (stitchBtn) {
        stitchBtn.style.display = 'inline-flex';
    }
    
    console.log('Final clipsGrid HTML:', clipsGrid.innerHTML.substring(0, 500)); // Show first 500 chars
}

// Create clip card
function createClipCard(clip) {
    console.log('createClipCard called with:', clip);
    console.log('clip.index:', clip.index, 'type:', typeof clip.index);
    console.log('clip.start:', clip.start, 'type:', typeof clip.start);
    console.log('clip.score:', clip.score, 'type:', typeof clip.score);
    
    const card = document.createElement('div');
    card.className = 'clip-card';
    card.style.animationDelay = `${clip.index * 0.1}s`;
    
    // Ensure valid start/end times
    const start = parseFloat(clip.start) || 0;
    const end = parseFloat(clip.end) || (start + (parseFloat(clip.duration) || 5));
    const duration = parseFloat(clip.duration) || (end - start);
    const score = Math.round(parseFloat(clip.score) || 50);
    
    console.log('Parsed values:', {start, end, duration, score}); // Debug
    
    card.innerHTML = `
        <div class="clip-header">
            <div class="clip-title">
                <i class="fas fa-video"></i> Clip #${clip.index}
            </div>
        </div>
        <div class="clip-info">
            <p><strong>Time:</strong> ${formatTime(start)} - ${formatTime(end)}</p>
            <p><strong>Duration:</strong> ${duration.toFixed(1)}s</p>
            <p><strong>Viral Potential:</strong> ${clip.reason || 'High engagement potential'}</p>
        </div>
        <div class="clip-actions">
            <button class="btn-preview" data-start="${start}" data-end="${end}">
                <i class="fas fa-play"></i> Preview
            </button>
            <button class="btn-download" data-jobid="${currentJobId}" data-filename="${clip.file}">
                <i class="fas fa-download"></i> Download
            </button>
        </div>
    `;
    
    // Attach event listeners instead of inline onclick
    const previewBtn = card.querySelector('.btn-preview');
    const downloadBtn = card.querySelector('.btn-download');
    
    previewBtn.addEventListener('click', () => {
        const s = parseFloat(previewBtn.dataset.start);
        const e = parseFloat(previewBtn.dataset.end);
        console.log('Preview button clicked:', {s, e});
        previewClip(s, e);
    });
    
    downloadBtn.addEventListener('click', () => {
        const jobId = downloadBtn.dataset.jobid;
        const filename = downloadBtn.dataset.filename;
        console.log('Download button clicked:', {jobId, filename});
        downloadClip(jobId, filename);
    });
    
    return card;
}

// Format time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Simulate AI analysis
async function simulateAnalysis() {
    const stages = [
        { progress: 20, text: 'Loading video...' },
        { progress: 40, text: 'Analyzing audio peaks...' },
        { progress: 60, text: 'Detecting motion intensity...' },
        { progress: 80, text: 'Identifying hook moments...' },
        { progress: 100, text: 'Generating clips...' }
    ];
    
    for (const stage of stages) {
        await new Promise(resolve => setTimeout(resolve, 800));
        progressFill.style.width = stage.progress + '%';
        progressText.textContent = stage.progress + '%';
        loading.querySelector('p').textContent = stage.text;
    }
}

// Generate results
function generateResults() {
    clipsGrid.innerHTML = '';
    
    const hookWords = document.getElementById('hookWords').value.split(',').map(w => w.trim()).filter(w => w);
    
    for (let i = 1; i <= clipCount; i++) {
        const startTime = Math.floor(Math.random() * 120);
        const endTime = startTime + selectedDuration;
        const viralScore = Math.floor(Math.random() * 30 + 70); // 70-100
        const reason = getRandomReason(hookWords);
        
        const clipCard = createClipCard(i, startTime, endTime, viralScore, reason);
        clipsGrid.appendChild(clipCard);
    }
}

// Get random reason for viral potential
function getRandomReason(customWords) {
    const defaultReasons = [
        'High energy moment detected',
        'Intense facial expressions',
        'Peak audio levels',
        'Rapid motion detected',
        'Engaging storytelling',
        'Emotional peak detected',
        'Action-packed sequence',
        'Hook words identified'
    ];
    
    const reasons = customWords.length > 0 
        ? [...defaultReasons, ...customWords.map(w => `"${w}" detected`)]
        : defaultReasons;
    
    return reasons[Math.floor(Math.random() * reasons.length)];
}

// Format time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Preview clip
function previewClip(startTime, endTime) {
    // Validate start and end times
    if (!startTime || isNaN(startTime) || startTime < 0) {
        startTime = 0;
    }
    if (!endTime || isNaN(endTime) || endTime <= startTime) {
        console.warn('Invalid end time, playing from start');
        endTime = startTime + 5; // Default 5 second preview
    }
    
    previewVideo.currentTime = startTime;
    previewVideo.play();
    
    // Stop at end time
    const checkTime = setInterval(() => {
        if (previewVideo.currentTime >= endTime) {
            previewVideo.pause();
            clearInterval(checkTime);
        }
    }, 100);
    
    // Scroll to video
    previewVideo.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Make functions globally accessible for onclick handlers
window.previewClip = previewClip;

// Download clip
async function downloadClip(jobId, clipFilename) {
    try {
        showNotification(`Preparing ${clipFilename} for download...`, 'info');
        
        const response = await fetch(`${API_URL}/download/${jobId}/${clipFilename}`);
        
        if (!response.ok) throw new Error('Download failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `clipcatch_${clipFilename}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification(`${clipFilename} downloaded successfully!`, 'success');
    } catch (error) {
        showNotification('Download failed!', 'error');
        console.error('Download error:', error);
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    
    let bgColor, icon;
    switch(type) {
        case 'success':
            bgColor = 'var(--success-color)';
            icon = 'check-circle';
            break;
        case 'error':
            bgColor = '#ef4444';
            icon = 'exclamation-circle';
            break;
        default:
            bgColor = 'var(--primary-color)';
            icon = 'info-circle';
    }
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${bgColor};
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    notification.innerHTML = `
        <i class="fas fa-${icon}"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Stitch all clips into one video
if (stitchBtn) {
    stitchBtn.addEventListener('click', async () => {
        if (!currentJobId) {
            showNotification('No clips available to stitch', 'error');
            return;
        }
        
        stitchBtn.disabled = true;
        stitchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stitching...';
        showNotification('Stitching all clips together...', 'info');
        
        try {
            const response = await fetch(`${API_URL}/stitch/${currentJobId}`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Stitch failed');
            }
            
            const data = await response.json();
            showNotification('Clips stitched successfully!', 'success');
            
            // Download the stitched video
            const downloadResponse = await fetch(`${API_URL}/download/${currentJobId}/${data.filename}`);
            if (!downloadResponse.ok) throw new Error('Download failed');
            
            const blob = await downloadResponse.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `clipcatch_stitched_${currentJobId}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showNotification('Stitched video downloaded!', 'success');
            
        } catch (error) {
            showNotification(`Stitch failed: ${error.message}`, 'error');
            console.error('Stitch error:', error);
        } finally {
            stitchBtn.disabled = false;
            stitchBtn.innerHTML = '<i class="fas fa-film"></i> Stitch All Clips';
        }
    });
}

// Initialize
console.log('ClipCatch AI initialized successfully!');
