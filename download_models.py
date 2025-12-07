# ===== FILE: download_models.py =====
#!/usr/bin/env python3
"""
Download all required models before first use.
Respects config.yaml paths.
"""

import os
import yaml
import logging
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_whisper(model_size, path, device):
    logger.info(f"Downloading Whisper model: {model_size}")

    ensure_dir(path)

    model = whisper.load_model(model_size, device=device)

    logger.info(f"✓ Whisper model saved in cache (local path: {path})")
    return model


def download_summarizer(model_name, path):
    logger.info(f"Downloading summarizer model: {model_name}")

    ensure_dir(path)

    AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=path)

    logger.info(f"✓ Summarizer downloaded → {path}")


def download_diarization(model_name):
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        logger.warning("⚠ Skipping diarization download — no HuggingFace token found.")
        return

    logger.info("Downloading diarization model (pyannote)...")

    try:
        from pyannote.audio import Pipeline
        Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        logger.info("✓ Diarization model downloaded")
    except Exception as e:
        logger.error(f"❌ Failed to download diarization model: {e}")


def download_models():
    config = load_config()  # may raise if config missing — desired behavior
    models_cfg = config.get("models", {})
    processing_cfg = config.get("processing", {})
    perf_cfg = config.get("performance", {})

    # Allow skipping heavy downloads (useful for Docker builds)
    if os.getenv("SKIP_MODEL_DOWNLOAD", "false").lower() in ("1", "true", "yes"):
        logger.info("SKIP_MODEL_DOWNLOAD is set → skipping model downloads.")
        return

    # Models
    whisper_size = models_cfg.get("whisper_size", "medium")
    summarizer_name = models_cfg.get("summarizer", "facebook/bart-large-cnn")
    diarization_name = models_cfg.get("diarization_model", "pyannote/speaker-diarization")

    # Paths: support both new config key 'paths' and older 'paths_dir' / defaults
    paths_cfg = config.get("paths", {})
    models_base = paths_cfg.get("models_dir") or config.get("models_dir") or "models"

    whisper_dir = os.path.join(models_base, "whisper")
    summarizer_dir = os.path.join(models_base, "summarizer")

    # Create model directories
    ensure_dir(whisper_dir)
    ensure_dir(summarizer_dir)

    # Device selection
    device_pref = perf_cfg.get("device", "auto")
    device = "cuda" if torch.cuda.is_available() and device_pref != "cpu" else "cpu"

    logger.info("==== Downloading Required Models ====")
    logger.info(f"Device: {device} | Whisper size: {whisper_size} | Summarizer: {summarizer_name}")

    # Whisper: uses whisper.load_model with download_root to control cache location
    try:
        logger.info(f"Downloading/Loading Whisper model '{whisper_size}' into {whisper_dir} ...")
        # whisper.load_model accepts download_root (alias may vary by version); use download_root where supported
        # fallback: set environment var WHISPER_CACHE or rely on default cache
        whisper.load_model(whisper_size, device=device, download_root=whisper_dir)
        logger.info("✓ Whisper ready")
    except TypeError:
        # older whisper versions may not accept download_root param; call normally and warn
        logger.warning("whisper.load_model does not accept download_root on this version; loading into default cache.")
        whisper.load_model(whisper_size, device=device)
        logger.info("✓ Whisper ready (cached in default location)")

    # Summarizer: download tokenizer + model into summarizer_dir (HF cache_dir)
    try:
        logger.info(f"Downloading summarizer '{summarizer_name}' into {summarizer_dir} ...")
        BartTokenizer.from_pretrained(summarizer_name, cache_dir=summarizer_dir)
        BartForConditionalGeneration.from_pretrained(summarizer_name, cache_dir=summarizer_dir)
        logger.info("✓ Summarizer ready")
    except Exception as e:
        logger.error(f"Failed to download summarizer '{summarizer_name}': {e}")
        raise

    # Optional: diarization (only if explicitly enabled)
    if processing_cfg.get("enable_diarization", False):
        hf_token = load_hf_token()
        if not hf_token:
            logger.warning("Diarization enabled in config but no HuggingFace token found — skipping diarization download.")
        else:
            try:
                logger.info(f"Downloading diarization model '{diarization_name}' ...")
                Pipeline.from_pretrained(diarization_name, use_auth_token=hf_token)
                logger.info("✓ Diarization model ready")
            except Exception as e:
                logger.error(f"Failed to download diarization model: {e}")
                logger.warning("Continuing without diarization model.")

    logger.info("\nAll models processed. Local model base dir: %s", models_base)


if __name__ == "__main__":
    download_models()
