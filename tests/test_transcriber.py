# ===== FIXED FILE: tests/test_transcriber.py =====
import pytest
import numpy as np
import wave
from pathlib import Path

from modules.transcriber import SpeechTranscriber


@pytest.fixture
def transcriber(tmp_path):
    """
    Create transcriber using tiny Whisper model for fast testing.
    Requires models to be pre-downloaded into models/whisper.
    """
    return SpeechTranscriber(
        model_size="tiny",
        model_dir="models/whisper",
        device="cpu",
        use_fp16=False
    )


def create_dummy_audio(tmp_path, duration=1.0, sample_rate=16000):
    """Create a simple WAV file with random noise."""
    samples = int(duration * sample_rate)
    audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)

    audio_path = tmp_path / "dummy.wav"

    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return audio_path


def test_transcriber_initialization(transcriber):
    """Verify model loads correctly."""
    assert transcriber.model is not None
    assert transcriber.device in ["cpu", "cuda"]


def test_transcribe_dummy_audio(tmp_path, transcriber):
    """Transcribe a dummy audio file (output may be empty for noise)."""
    audio_path = create_dummy_audio(tmp_path)

    result = transcriber.transcribe(str(audio_path))

    assert isinstance(result, dict)
    assert "text" in result
    assert "segments" in result
    assert "language" in result
    assert isinstance(result["segments"], list)


def test_transcriber_get_segments(tmp_path, transcriber):
    """Test segment extraction method."""
    audio_path = create_dummy_audio(tmp_path)

    segments = transcriber.get_segments(str(audio_path))

    assert isinstance(segments, list)
    for seg in segments:
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg


def test_word_timestamps_placeholder(tmp_path, transcriber):
    """Test placeholder word timestamp method."""
    audio_path = create_dummy_audio(tmp_path)

    segments = transcriber.word_timestamps(str(audio_path))

    assert isinstance(segments, list)
