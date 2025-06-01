from __future__ import annotations

import os

# Set environment variables to handle MPS compatibility issues
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Optionally, you can also completely disable MPS if needed:
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from dataclasses import dataclass
from typing import List, Tuple

from peft import PeftModel
from rapidfuzz import fuzz
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from scripts.loggers import get_logger

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class SegmenterConfig:
    model_name: str = "jinaai/text-seg-lm-qwen2-0.5b-cot-topic-chunking"
    base_model_name: str = (
        "Qwen/Qwen2-0.5B-Instruct"  # Base model for PEFT adapter (non-quantized)
    )
    window_chars: int = 3000  # ≈ 1000 tokens in English prose
    stride_chars: int = 2700  # 10 % overlap
    max_new_tokens: int = 256
    fuzzy_threshold: int = 90  # minimum RapidFuzz ratio for head ↔ text align


# ────────────────────────────────────────────────────────────────────────────────
# Segmenter class (driver)
# ────────────────────────────────────────────────────────────────────────────────


class TopicSegmenter:
    """Window-based driver around the topic-qwen-0.5b model."""

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name, trust_remote_code=True
        )

        # Load base model first
        logger.info(f"Loading base model: {cfg.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_name, device_map="auto"
        )

        # Load PEFT adapter on top of base model
        logger.info(f"Loading PEFT adapter: {cfg.model_name}")
        self.model = PeftModel.from_pretrained(base_model, cfg.model_name)

        logger.info(
            f"Model loaded: {cfg.model_name} (adapter) on {cfg.base_model_name} (base) with device map: {self.model.base_model.hf_device_map}"
        )

    # --- internal helpers ---
    def _slm_heads(self, chunk: str) -> List[str]:
        toks = self.tokenizer(chunk, return_tensors="pt", truncation=True)
        device = next(self.model.parameters()).device
        toks = {k: v.to(device) for k, v in toks.items()}

        gen = self.model.generate(**toks, max_new_tokens=self.cfg.max_new_tokens)

        # Use full output since this model reformats with headers embedded
        full_output = self.tokenizer.decode(gen[0], skip_special_tokens=True)

        # Extract headers (lines starting with #)
        import re

        headers = re.findall(r"^#+\s+(.+)$", full_output, re.MULTILINE)

        return headers

    @staticmethod
    def _find_with_fuzz(head: str, haystack: str, threshold: int) -> int | None:
        """Return index where *head* matches *haystack* with >= threshold; else None."""
        # quick exact check
        idx = haystack.find(head)
        if idx != -1:
            return idx
        # fuzzy sliding window (wide-net, but _haystack_ is only one window stride)
        span = max(20, len(head))
        for offs in range(0, len(haystack) - span, 20):
            cand = haystack[offs : offs + span]
            if fuzz.ratio(head, cand) >= threshold:
                return offs
        return None

    # --- public API ---
    def segment(self, text: str) -> List[Tuple[int, int, str]]:
        """Return list of (start, end, head) tuples defining the partition."""
        cfg = self.cfg
        heads: List[str] = []
        for left in range(0, len(text), cfg.stride_chars):
            chunk = text[left : left + cfg.window_chars]
            heads.extend(self._slm_heads(chunk))

        # align heads to original text
        positions: List[int] = []
        cursor = 0
        for h in heads:
            # search up to 40 chars for speed, fallback fuzzy
            key = h[:40]
            rel = text[cursor:].find(key)
            if rel != -1:
                pos = cursor + rel
            else:
                rel = self._find_with_fuzz(key, text[cursor:], cfg.fuzzy_threshold)
                if rel is None:
                    continue  # skip head, overlap should cover lost boundary
                pos = cursor + rel
            if positions and pos <= positions[-1]:
                continue  # duplicate or backward, ignore
            positions.append(pos)
            cursor = pos

        # always append terminal boundary
        positions.append(len(text))

        # build segments
        segments = [
            (positions[i], positions[i + 1], heads[i] if i < len(heads) else "<tail>")
            for i in range(len(positions) - 1)
        ]
        return segments
