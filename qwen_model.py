"""
Qwen model and tokenizer loading with caching.
"""
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEFAULT_QWEN_PATH

_QWEN_TOKENIZER: Optional[AutoTokenizer] = None
_QWEN_MODEL: Optional[AutoModelForCausalLM] = None


def load_qwen(
    model_path: str | None = None,
    device: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load Qwen model and tokenizer. On CPU uses float32; on CUDA uses bf16 and
    device_map='auto'. Returns (tokenizer, model).
    """
    global _QWEN_TOKENIZER, _QWEN_MODEL
    if _QWEN_TOKENIZER is not None and _QWEN_MODEL is not None:
        return _QWEN_TOKENIZER, _QWEN_MODEL

    path = model_path or DEFAULT_QWEN_PATH
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. PyTorch is not seeing your GPU. "
            "Install the CUDA build of PyTorch (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121) "
            "or use the CPU branch (uncomment in qwen_model.py)."
        )
        
    if torch.cuda.is_available():
        # GPU: explicit cuda:0
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
    else:
        # CPU fallback (PyTorch is CPU-only or CUDA not visible)
        warnings.warn(
            "CUDA not available — running on CPU. For GPU, install PyTorch with CUDA, e.g.: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to("cpu")

    _QWEN_TOKENIZER = tokenizer
    _QWEN_MODEL = model
    return tokenizer, model
