"""
Full QA pipeline: retrieve snippets, build prompt, run Qwen.
"""
from typing import Any, Dict

import torch

from config import get_device
from qa_prompt import build_prompt
from qa_retrieval import build_context_snippets
from qwen_model import load_qwen


def answer_query(
    query: str,
    model_path: str | None = None,
    top_k: int = 5,
    max_new_tokens: int = 512,
    device: str | None = None,
) -> Dict[str, Any]:
    """
    Full QA pipeline: retrieve top_k pages, build grounded prompt,
    generate answer with Qwen using only the given sources.
    Returns dict with keys: answer, snippets, prompt.
    """
    if not query.strip():
        return {"answer": "", "snippets": [], "prompt": ""}

    snippets = build_context_snippets(query, top_k=top_k)
    prompt = build_prompt(query, snippets)
    effective_device = device or get_device()
    tokenizer, model = load_qwen(model_path, device=effective_device)

    messages = [
        {
            "role": "system",
            "content": "أنت مساعد لغوي خبير في الفلسفة العربية الكلاسيكية.",
        },
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = outputs[0, inputs["input_ids"].size(1) :]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "answer": answer.strip(),
        "snippets": snippets,
        "prompt": prompt,
    }
