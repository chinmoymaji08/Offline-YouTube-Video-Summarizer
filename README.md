# Offline YouTube Video Summarizer
An end-to-end offline system that downloads a YouTube video, extracts audio, transcribes speech using Whisper, and produces a concise summary using a local summarization model. Everything runs completely offline — no cloud APIs.

## 1. Project Overview
This project builds a fully offline pipeline that converts a YouTube video into a readable text summary.
It performs four main tasks:

1. Download audio from a YouTube URL  
2. Transcribe speech into text using an offline Whisper model  
3. Summarize the transcript using an offline transformer model  
4. Serve results through a simple web interface (Flask)

The system is designed for environments where internet access is restricted or cloud-based APIs cannot be used due to compliance or cost constraints.

## 2. Setup and Installation Instructions

### 2.1 Requirements
- Python 3.10+
- FFmpeg installed locally
- 8–16 GB RAM recommended
- Optional: CUDA GPU
- Docker Desktop (if using Docker)

### 2.2 Local Setup
```bash
git clone <your-repo>
cd Offline-YouTube-Summarizer
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python app.py
```
Open: http://localhost:5000

### 2.3 Docker Setup
```bash
docker-compose build
docker-compose up
```

## 3. Design Choices & Justification

### Whisper (Medium)
Chosen for:
- Strong accuracy  
- Fully offline  
- Balanced size/performance  

### BART Large CNN
Chosen for:
- High‑quality abstractive summaries  
- Works well with long conversational text  

### Chunking System
BART cannot handle very long sequences, so transcripts are:
- Split into chunks  
- Summarized individually  
- Combined and re‑summarized  

This improves accuracy and prevents memory issues.

## 4. Usage Instructions

### Web UI
Open:
```
http://localhost:5000
```

Enter a YouTube URL → Generate Summary.

### CLI Usage
```bash
python summarize.py "<YOUTUBE_URL>" --max-length 150
```

### API Usage
POST `/api/summarize`
```json
{
  "url": "https://www.youtube.com/watch?v=123",
  "maxLength": 150,
  "diarize": false
}
```

## 5. Challenges Faced

### PyAnnote + Torch Dependency Issues
Resolved by pinning compatible versions.

### Large Model Downloads in Docker
Moved model download step outside Docker build.

### Whisper RAM Usage
Implemented device auto-selection and optional smaller models.

### Long Transcripts Exceeding Token Limits
Implemented multi-step chunk summarization.

## 6. Project Structure
```
youtube-video-summarizer/
├── app.py                         # Flask web application
├── summarize.py                   # CLI summarizer tool
├── config.yaml                    # Main configuration file
├── requirements.txt               # Python dependencies
├── download_models.py             # Offline model downloader
├── docker-compose.yml             # Docker services
├── Dockerfile                     # Build image
├── pytest.ini                     # Pytest config
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── downloader.py              # YouTube audio downloader
│   ├── transcriber.py             # Whisper STT engine
│   ├── summarizer.py              # BART summarization engine
│   ├── diarizer.py                # Optional speaker diarization
│   └── utils.py
│
├── templates/
│   └── index.html                 # Web UI
│
├── models/                        # Offline cached models
│   ├── whisper/
│   └── summarizer/
│
├── docs/
│   ├── ARCHITECTURE.png           # System architecture diagram
│   └── demo.mp4                   # Working demo video

```

## 7. Future Improvements
- Add diarization  
- Use faster Whisper inference frameworks  
- Add GPU-enabled Docker images  
- Improve long-summary coherence  

