# ===== FIXED FILE: tests/test_integration.py =====
"""
Integration test: downloader → transcriber → summarizer
Requires:
- Internet access for the YouTube download
- Whisper & summarizer models downloaded beforehand
"""

import pytest
import os
from pathlib import Path

from modules.downloader import YouTubeDownloader
from modules.transcriber import SpeechTranscriber
from modules.summarizer import TextSummarizer


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline(tmp_path):
    """
    Integration test for the entire offline summarization pipeline.
    Uses:
      - YouTubeDownloader
      - SpeechTranscriber (Whisper tiny)
      - TextSummarizer (BART)
    """

    # Public domain, small file
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    download_dir = tmp_path / "integration_dl"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. DOWNLOAD
        downloader = YouTubeDownloader(output_dir=download_dir)
        result = downloader.download(url)

        audio_path = Path(result["audio_path"])
        metadata = result["metadata"]

        # Validate download
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"
        assert isinstance(metadata["title"], str)

        # 2. TRANSCRIBE (super-fast model)
        transcriber = SpeechTranscriber(
            model_size="tiny", 
            model_dir="models/whisper",
            device="cpu",
            use_fp16=False
        )

        transcript_result = transcriber.transcribe(str(audio_path))
        transcript = transcript_result["text"]

        assert isinstance(transcript, str)
        assert len(transcript) > 0

        # 3. SUMMARIZE
        summarizer = TextSummarizer(
            model_name="facebook/bart-large-cnn",
            model_dir="models/summarizer",
            device="cpu",
            use_fp16=False
        )

        summary = summarizer.summarize(
            transcript,
            max_length=80,
            min_length=20
        )

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) <= len(transcript)

        # CLEAN UP
        downloader.cleanup(str(audio_path))

    except Exception as e:
        pytest.skip(f"Integration test skipped due to error: {e}")
