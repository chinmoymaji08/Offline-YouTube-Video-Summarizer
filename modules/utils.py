# ===== FILE: modules/utils.py =====
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


# ------------------------------
# Logging
# ------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging (only once)."""
    logger = logging.getLogger("yt_summarizer")
    if not logger.handlers:  # Prevent duplicate handlers
        level = logging.DEBUG if verbose else logging.INFO
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger


# ------------------------------
# Configuration
# ------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration file regardless of working directory."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------
# Directories
# ------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------
# Device and Performance
# ------------------------------

def get_device(config: Dict[str, Any]) -> str:
    """Determine computation device (cpu/cuda)."""
    import torch
    pref = config["performance"]["device"]

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cpu"  # Fallback


def safe_fp16(device: str, fp16_flag: bool) -> bool:
    """Enable FP16 only when device is CUDA."""
    return fp16_flag and device == "cuda"


# ------------------------------
# Helpers
# ------------------------------

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters and trim whitespace."""
    invalid = '<>:"/\\|?*'
    for c in invalid:
        filename = filename.replace(c, "_")
    filename = filename.strip()  # extra fix
    return filename[:200]


def load_hf_token() -> str | None:
    """Load HuggingFace token from .env or environment."""
    load_dotenv()
    return os.getenv("HUGGINGFACE_TOKEN", None)


def file_size_mb(path: str | Path) -> float:
    """Return file size in MB."""
    try:
        size = os.path.getsize(path) / (1024 * 1024)
        return round(size, 2)
    except:
        return 0.0
