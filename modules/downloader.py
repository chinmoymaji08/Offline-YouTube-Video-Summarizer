# ===== FILE: modules/downloader.py =====
import os
import uuid
import logging
from pathlib import Path
from typing import Dict
import yt_dlp

from .utils import sanitize_filename, ensure_dir

logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """
    Downloads YouTube audio safely and converts to 16kHz mono WAV for Whisper.
    """

    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = ensure_dir(output_dir)

    def download(self, url: str, audio_format: str = "wav") -> Dict[str, str]:
        """
        Download and convert YouTube audio.

        Returns:
            {
                "audio_path": "<path>",
                "metadata": {...}
            }
        """
        logger.info(f"Downloading audio from: {url}")

        # Unique file prefix to avoid collisions
        unique_id = uuid.uuid4().hex[:8]

        # yt-dlp template WITHOUT raw title to avoid invalid filenames
        outtmpl = str(self.output_dir / f"yt_{unique_id}.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,

            # Extract only audio
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                    "preferredquality": "192",
                }
            ],

            # Ensure Whisper-compatible parameters
            "postprocessor_args": [
                "-ar", "16000",  # Sample rate
                "-ac", "1"       # Mono
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                raw_title = info.get("title", "audio")
                safe_title = sanitize_filename(raw_title)

                # yt-dlp final output filename after postprocessing
                final_path = Path(self.output_dir / f"yt_{unique_id}.{audio_format}")

                if not final_path.exists():
                    raise FileNotFoundError(
                        f"yt-dlp finished but output file not found: {final_path}"
                    )

                metadata = {
                    "title": safe_title,
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "upload_date": info.get("upload_date", "Unknown"),
                }

                logger.info(f"Downloaded: {safe_title} ({metadata['duration']}s)")
                return {
                    "audio_path": str(final_path),
                    "metadata": metadata
                }

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise RuntimeError(f"Failed to download video: {str(e)}")

    def cleanup(self, audio_path: str):
        """Remove downloaded audio file"""
        try:
            p = Path(audio_path)
            if p.exists():
                p.unlink()
                logger.info(f"Cleaned up: {audio_path}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
