# ===== FILE: modules/diarizer.py =====
import logging
from typing import List, Dict
import torch
import os
if os.getenv("DISABLE_DIARIZATION", "false").lower() == "true":
    DIARIZATION_AVAILABLE = False


from .utils import load_hf_token

logger = logging.getLogger(__name__)

# Attempt pyannote import safely
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False


class SpeakerDiarizer:
    """
    Offline speaker diarization wrapper using pyannote.audio 2.x.
    Requires a HuggingFace access token.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization",
        device: str = "cpu"
    ):
        if not DIARIZATION_AVAILABLE:
            raise RuntimeError(
                "pyannote.audio is not installed. "
                "Install with: pip install pyannote.audio==2.1.1"
            )

        self.device = device
        self.model_name = model_name

        # Load HF access token
        self.hf_token = load_hf_token()
        if not self.hf_token:
            raise RuntimeError(
                "HuggingFace token not found. "
                "Set HUGGINGFACE_TOKEN in your .env file."
            )

        logger.info(f"Loading pyannote diarization model: {model_name}")

        try:
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=self.hf_token
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization model: {e}")

        # Move to desired device (if supported)
        try:
            self.pipeline.to(self.device)
        except Exception:
            logger.warning("Pipeline device placement not supported. Using default device.")

        logger.info("Diarization model loaded successfully.")

    # -----------------------------------
    # Perform diarization
    # -----------------------------------

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run diarization and return speaker segments:
        [
            { "start": float, "end": float, "speaker": str }
        ]
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Running diarization → {audio_path}")

        try:
            diarization_result = self.pipeline(audio_path)
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")

        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })

        if not segments:
            logger.warning("No speaker segments detected.")
        else:
            logger.info(f"Detected {len(set(s['speaker'] for s in segments))} speakers.")

        return segments

    # -----------------------------------
    # Merge transcript + diarization
    # -----------------------------------

    def merge_with_transcript(
        self,
        transcript_segments: List[Dict],
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """
        Adds 'speaker' labels to Whisper transcript segments based on temporal overlap.
        """

        merged = []

        for t in transcript_segments:
            t_start = float(t.get("start", 0))
            t_end = float(t.get("end", 0))

            best_speaker = "Unknown"
            max_overlap = 0

            for d in diarization_segments:
                d_start = d["start"]
                d_end = d["end"]

                # Calculate overlap interval
                overlap = max(0, min(t_end, d_end) - max(t_start, d_start))

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d["speaker"]

            merged.append({
                **t,  # keep "start", "end", "text"
                "speaker": best_speaker
            })

        return merged
