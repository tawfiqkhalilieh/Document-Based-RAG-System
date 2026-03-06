"""
Prompt construction for grounded QA (Arabic philosophy).
"""
from typing import Any, Dict, List

SYSTEM_INSTRUCTIONS = (
    "أنت باحث متخصص في الفلسفة العربية الكلاسيكية.\n"
    "يجب أن تعتمد حصريًا على المقاطع المعطاة في قسم (المصادر) للإجابة.\n"
    "إذا لم تجد جوابًا واضحًا في المصادر، فقل بوضوح إن الجواب غير متوفر في المواد المعطاة.\n"
    "اكتب إجابتك بالعربية الفصيحة، واذكر المراجع في آخر الجواب بصيغة (اسم الكتاب، رقم الصفحة) عند الاقتضاء."
)


def build_prompt(query: str, snippets: List[Dict[str, Any]]) -> str:
    """
    Build a grounded prompt instructing the model to answer only from
    the provided passages. Uses Arabic system instructions and source blocks.
    """
    context_blocks: List[str] = []
    for s in snippets:
        header = f"المصدر: {s['book']}، صفحة {s['page']}"
        body = (s.get("text") or "").strip()
        if not body:
            continue
        block = f"{header}\n--------------------\n{body}"
        context_blocks.append(block)

    context_text = (
        "\n\n\n".join(context_blocks) if context_blocks else "لا توجد مقاطع ذات صلة متاحة."
    )

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"المصادر:\n\n"
        f"{context_text}\n\n"
        f"السؤال:\n{query}\n\n"
        "أعطِ جوابًا موجزًا ودقيقًا معتمدًا فقط على هذه المصادر."
    )
    return prompt
