# ===== FILE: modules/transcriber.py =====
import logging
import os
from typing import Dict, Optional, List
from pathlib import Path

import torch
import whisper

from .utils import safe_fp16, ensure_dir

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """
    Offline Whisper-based speech-to-text transcriber.
    Loads models from local directory to support offline mode.
    """

    def __init__(
        self,
        model_size: str = "medium",
        model_dir: str = "models/whisper",
        device: str = "cpu",
        use_fp16: bool = False
    ):
        self.model_size = model_size
        self.device = device
        self.use_fp16 = safe_fp16(device, use_fp16)

        # Ensure local model dir exists
        self.model_dir = Path(model_dir)
        ensure_dir(self.model_dir)

        logger.info(f"Loading Whisper '{model_size}' model (fp16={self.use_fp16})")

        # Force loading from local model directory
        self.model = whisper.load_model(
            name=model_size,
            device=self.device,
            download_root=str(self.model_dir)
        )

        logger.info("Whisper model loaded successfully.")

    # ----------------------------
    # Core Transcription
    # ----------------------------

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio offline using Whisper.

        Returns:
            {
                'text': str,
                'segments': list,
                'language': str
            }
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio → {audio_path}")

        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=self.use_fp16,
                verbose=False,
                condition_on_previous_text=False,  # prevents hallucinations
                temperature=0  # more stable for long content
            )

            transcript = result.get("text", "").strip()
            segments = result.get("segments", [])
            lang = result.get("language", "unknown")

            logger.info(f"Transcription completed. Characters: {len(transcript)}")

            return {
                "text": transcript,
                "segments": segments,
                "language": lang
            }

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription error: {e}")

    # ----------------------------
    # Cleaned Timestamps (Segment-level)
    # ----------------------------

    def get_segments(self, audio_path: str) -> List[Dict]:
        """Return segment-level timestamps only."""
        result = self.transcribe(audio_path)
        formatted = []
        for seg in result["segments"]:
            formatted.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip()
            })
        return formatted

    # ----------------------------
    # Placeholder for word timestamps (WhisperX)
    # ----------------------------

    def word_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Placeholder for WhisperX integration.
        Returns segment-level timestamps until WhisperX is integrated.
        """
        logger.warning("word_timestamps() called — WhisperX not integrated yet.")
        return self.get_segments(audio_path)
