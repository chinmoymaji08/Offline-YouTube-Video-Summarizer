# ===== FILE: modules/summarizer.py =====
import logging
from typing import List
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import ensure_dir, safe_fp16

logger = logging.getLogger(__name__)


class TextSummarizer:
    """
    Offline summarizer using BART or any HF seq2seq model.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        model_dir: str = "models/summarizer",
        device: str = "cpu",
        use_fp16: bool = False
    ):
        self.device = device
        self.use_fp16 = safe_fp16(device, use_fp16)

        self.model_dir = Path(model_dir)
        ensure_dir(self.model_dir)

        logger.info(f"Loading summarizer '{model_name}' on {self.device} (fp16={self.use_fp16})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.model_dir
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=self.model_dir
        )

        self.model.to(self.device)

        if self.use_fp16:
            self.model.half()

        logger.info("Summarizer loaded successfully.")

    # ----------------------------------
    # Public summarize() function
    # ----------------------------------

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
        chunk_size: int = 1024
    ) -> str:

        logger.info(f"Summarizing text ({len(text)} chars)")

        tokens = self.tokenizer.encode(text, truncation=False)

        if len(tokens) <= chunk_size:
            summary = self._summarize_single(text, max_length, min_length)
            return summary

        # Otherwise use chunking
        logger.info(f"Text too long ({len(tokens)} tokens). Using chunking...")
        return self._summarize_long(text, max_length, min_length, chunk_size)

    # ----------------------------------
    # Single chunk summarization
    # ----------------------------------

    def _summarize_single(self, text: str, max_length: int, min_length: int) -> str:
        """Summarize text that fits into the model context window."""

        inputs = self.tokenizer(
            text,
            max_length=900,  # Safe for BART
            truncation=True,
            padding=False,
            return_tensors="pt"
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.8,
            early_stopping=True,
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ----------------------------------
    # Chunk-based long text summarization
    # ----------------------------------

    def _summarize_long(
        self,
        text: str,
        max_length: int,
        min_length: int,
        chunk_size: int
    ) -> str:
        """
        Split text into chunks, summarize each, then summarize the combined summaries.
        """

        # Split into sentences more reliably
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = []
        current_tokens = 0

        # Build chunks safely
        for sentence in sentences:
            t_len = len(self.tokenizer.encode(sentence, truncation=False))

            if current_tokens + t_len > chunk_size:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = t_len
            else:
                current.append(sentence)
                current_tokens += t_len

        if current:
            chunks.append(" ".join(current))

        logger.info(f"Created {len(chunks)} text chunks")

        # Summarize each chunk
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {idx + 1}/{len(chunks)}")

            # Keep chunk summaries meaningful
            chunk_summary = self._summarize_single(
                chunk,
                max_length=200,
                min_length=40
            )
            chunk_summaries.append(chunk_summary)

        combined = " ".join(chunk_summaries)
        combined_tokens = len(self.tokenizer.encode(combined, truncation=False))

        # Second pass summary if needed
        if combined_tokens > 900:
            logger.info("Re-summarizing combined summary (second pass)")
            final_summary = self._summarize_single(
                combined,
                max_length=max_length,
                min_length=min_length
            )
        else:
            final_summary = combined

        return final_summary
