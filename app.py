# ===== FILE: app.py =====
"""
Flask Web Application for Offline YouTube Summarizer
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import time
import os
from pathlib import Path

from modules.utils import load_config, get_device, ensure_dir
from modules.downloader import YouTubeDownloader
from modules.transcriber import SpeechTranscriber
from modules.summarizer import TextSummarizer
from modules.diarizer import SpeakerDiarizer

app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
device = get_device(config)

# Enforce FP16 only on CUDA
use_fp16 = config["performance"]["use_fp16"] if device == "cuda" else False

# Directories
DOWNLOADS_DIR = config["paths"]["downloads_dir"]
MODELS_DIR = config["paths"]["models_dir"]

ensure_dir(DOWNLOADS_DIR)
ensure_dir(MODELS_DIR)

# Lazy-loaded models
_transcriber = None
_summarizer = None
_diarizer = None


def get_transcriber():
    """Lazy load Whisper transcriber model"""
    global _transcriber
    if _transcriber is None:
        logger.info("Loading Whisper STT model...")
        _transcriber = SpeechTranscriber(
            model_size=config["models"]["whisper_size"],
            model_dir=os.path.join(MODELS_DIR, "whisper"),
            device=device,
            use_fp16=use_fp16
        )
    return _transcriber


def get_summarizer():
    """Lazy load summarizer model"""
    global _summarizer
    if _summarizer is None:
        logger.info("Loading summarizer model...")
        _summarizer = TextSummarizer(
            model_name=config["models"]["summarizer"],
            model_dir=os.path.join(MODELS_DIR, "summarizer"),
            device=device,
            use_fp16=use_fp16
        )
    return _summarizer


def get_diarizer():
    """Lazy load speaker diarization model"""
    global _diarizer
    if _diarizer is None:
        logger.info("Loading diarization model...")
        _diarizer = SpeakerDiarizer(
            model_name=config["models"]["diarization_model"],
            device=device
        )
    return _diarizer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()

        if not data or "url" not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data["url"]
        enable_diarize = data.get("diarize", False)
        max_length = data.get(
            "max_length",
            config["processing"]["max_summary_length"]
        )

        logger.info(f"Received request → {url}")

        start_time = time.time()

        # -------- Download --------
        downloader = YouTubeDownloader(DOWNLOADS_DIR)
        download_result = downloader.download(url)
        audio_path = download_result["audio_path"]
        metadata = download_result["metadata"]

        # -------- Transcribe --------
        transcriber = get_transcriber()
        transcript_data = transcriber.transcribe(audio_path)

        transcript = transcript_data["text"]
        segments = transcript_data["segments"]

        # -------- Optional diarization --------
        speakers = None
        transcript_for_summary = transcript

        if enable_diarize:
            try:
                diarizer = get_diarizer()
                diarization = diarizer.diarize(audio_path)
                merged = diarizer.merge_with_transcript(segments, diarization)

                # Format speaker transcript
                formatted = []
                current_speaker = None
                for seg in merged:
                    if seg["speaker"] != current_speaker:
                        current_speaker = seg["speaker"]
                        formatted.append(f"\n{current_speaker}:")
                    formatted.append(seg["text"])

                transcript_for_summary = " ".join(formatted)
                speakers = sorted(list({s["speaker"] for s in merged}))

            except Exception as e:
                logger.warning(f"Diarization failed, using plain transcript: {e}")
                transcript_for_summary = transcript

        # -------- Summarization --------
        summarizer = get_summarizer()
        summary = summarizer.summarize(
            transcript_for_summary,
            max_length=max_length,
            min_length=config["processing"]["min_summary_length"],
            chunk_size=config["processing"]["chunk_size"]
        )

        # -------- Cleanup --------
        if not config["output"]["save_audio"]:
            downloader.cleanup(audio_path)

        elapsed = time.time() - start_time

        # Limit transcript in API output to avoid freezing frontend
        full_transcript = transcript
        if len(full_transcript) > 15000:
            full_transcript = full_transcript[:15000] + "... [truncated]"

        return jsonify({
            "status": "success",
            "summary": summary,
            "transcript": full_transcript,
            "metadata": metadata,
            "speakers": speakers,
            "duration": elapsed,
            "processing_info": {
                "device": device,
                "fp16": use_fp16,
                "transcript_length": len(transcript),
                "summary_length": len(summary)
            }
        })

    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": device,
        "models_loaded": {
            "transcriber": _transcriber is not None,
            "summarizer": _summarizer is not None,
            "diarizer": _diarizer is not None
        }
    })


if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=config["server"]["debug"]
    )
