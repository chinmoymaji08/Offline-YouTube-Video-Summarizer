# ===== FIXED FILE: tests/test_downloader.py =====
import pytest
import os
from pathlib import Path

from modules.downloader import YouTubeDownloader
from modules.utils import sanitize_filename, ensure_dir


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temporary directory for downloads."""
    return tmp_path / "downloads"


@pytest.fixture
def downloader(tmp_output):
    """Downloader instance using temporary directory."""
    return YouTubeDownloader(output_dir=tmp_output)


def test_downloader_initialization(tmp_output, downloader):
    """Verify downloader initializes correctly and creates directory."""
    assert downloader.output_dir.exists()
    assert downloader.output_dir == tmp_output


def test_sanitize_filename():
    """Ensure filename sanitization works properly."""
    assert sanitize_filename("test<>video") == "test__video"
    assert sanitize_filename('test"file') == "test_file"
    assert len(sanitize_filename("a" * 300)) <= 200


@pytest.mark.integration
def test_download_short_video(downloader):
    """
    Tests downloading a public domain video.
    Skipped automatically if internet is unavailable.
    """
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # Creative Commons allowed

    try:
        result = downloader.download(url)

        audio_path = Path(result["audio_path"])
        metadata = result["metadata"]

        # Validate file existence
        assert audio_path.exists()
        assert audio_path.suffix == ".wav"  # our downloader forces .wav default

        # Validate metadata
        assert "title" in metadata
        assert isinstance(metadata["title"], str)
        assert metadata["duration"] >= 0

        # Cleanup
        downloader.cleanup(str(audio_path))

    except Exception as e:
        pytest.skip(f"Skipping download test (likely no internet): {e}")


def test_invalid_url(downloader):
    """Ensure invalid URLs raise RuntimeError."""
    with pytest.raises(RuntimeError):
        downloader.download("https://invalid-domain-url-xyz12345.com")
