# ===== FIXED FILE: tests/test_summarizer.py =====
import pytest
from modules.summarizer import TextSummarizer


@pytest.fixture(scope="session")
def summarizer():
    """
    Create summarizer instance using local model directory.
    Using session scope to avoid reloading BART-large-CNN for every test.
    """
    return TextSummarizer(
        model_name="facebook/bart-large-cnn",
        model_dir="models/summarizer",
        device="cpu",
        use_fp16=False
    )


def test_summarizer_initialization(summarizer):
    """Verify summarizer loads correctly."""
    assert summarizer.model is not None
    assert summarizer.tokenizer is not None
    assert summarizer.device == "cpu"


def test_summarize_short_text(summarizer):
    """Summarize short text without chunking."""
    text = (
        "Artificial intelligence is changing industries. Machine learning algorithms "
        "recognize patterns and make accurate predictions. Deep learning powers breakthroughs "
        "in vision and language models."
    )

    summary = summarizer.summarize(text, max_length=50, min_length=20)

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) <= len(text)


def test_summarize_long_text_chunking(summarizer):
    """Summarize long text that triggers chunking algorithm."""
    long_text = (
        "Artificial intelligence is transforming the world in unprecedented ways. "
        "Machine learning algorithms recognize complex patterns in vast datasets. "
        "Deep learning powers breakthroughs in computer vision, NLP, and speech recognition. "
        "AI is applied in healthcare, finance, transportation, and more. "
        "These capabilities raise ethical questions involving bias, privacy, and automation. "
    ) * 50  # Long enough to force chunking

    summary = summarizer.summarize(
        long_text,
        max_length=150,
        min_length=50,
        chunk_size=512
    )

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(long_text)


def test_chunk_summarization_direct_call():
    """
    Test the private _summarize_single logic.
    This is safe because the method is deterministic and useful to validate.
    """
    summarizer = TextSummarizer(
        model_name="facebook/bart-large-cnn",
        model_dir="models/summarizer",
        device="cpu",
        use_fp16=False
    )

    text = "This is a short test sentence for summarization."

    summary = summarizer._summarize_single(
        text,
        max_length=40,
        min_length=10
    )

    assert isinstance(summary, str)
    assert len(summary) > 0
