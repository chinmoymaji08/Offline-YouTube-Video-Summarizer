# ===== FIXED FILE: tests/test_utils.py =====
import pytest
import logging
from pathlib import Path

from modules.utils import (
    setup_logging,
    load_config,
    ensure_dir,
    format_time,
    sanitize_filename,
    safe_fp16,
    file_size_mb,
)


def test_setup_logging():
    """Ensure logger initializes correctly and at correct level."""
    logger = setup_logging(verbose=True)
    assert isinstance(logger, logging.Logger)
    assert logger.getEffectiveLevel() == logging.DEBUG

    logger2 = setup_logging(verbose=False)
    assert logger2.getEffectiveLevel() == logging.INFO


def test_ensure_dir(tmp_path):
    """Test safe directory creation."""
    test_dir = tmp_path / "created_dir"
    result = ensure_dir(test_dir)

    assert result.exists()
    assert result.is_dir()


def test_format_time():
    """Ensure time formatting works correctly."""
    assert format_time(0) == "00:00:00"
    assert format_time(61) == "00:01:01"
    assert format_time(3661) == "01:01:01"
    assert format_time(3723) == "01:02:03"


def test_sanitize_filename_basic():
    """Test removal of invalid filename characters."""
    assert sanitize_filename("normal_file.txt") == "normal_file.txt"
    assert sanitize_filename("file<with>bad:chars") == "file_with_bad_chars"
    assert sanitize_filename('file"with"quotes') == "file_with_quotes"
    assert sanitize_filename(" spaced name ").startswith("spaced")
    assert sanitize_filename(" spaced name ").endswith("name")


def test_sanitize_filename_length():
    """Ensure filenames longer than limit get trimmed."""
    long_name = "a" * 300
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 200


def test_safe_fp16():
    """Test FP16 logic."""
    # CPU → FP16 must be disabled
    assert safe_fp16("cpu", True) is False

    # CUDA → FP16 allowed
    assert safe_fp16("cuda", True) is True

    # If fp16_flag=False, always false
    assert safe_fp16("cuda", False) is False


def test_file_size_mb(tmp_path):
    """Test file size calculation."""
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"A" * 1048576)  # 1 MB

    size = file_size_mb(test_file)
    assert 0.9 <= size <= 1.1  # allow slight rounding error


def test_load_config_valid(tmp_path):
    """Test loading an actual config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test_value: 123")

    config = load_config(str(config_file))
    assert config["test_value"] == 123


def test_load_config_missing():
    """Ensure missing config file raises error."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
