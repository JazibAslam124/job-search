"""Step 3 of the pipeline: expand target job titles into synonyms/variants
so keyword search on job boards doesn't miss postings phrased differently.

This is intentionally a *small, bounded* LLM task (expansion, not query
syntax) — the actual API query construction stays deterministic and lives
in query_builder.py.
"""

import json

from groq import Groq

MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You expand a list of job titles into additional search \
variants that job boards might use instead. Include synonyms, common \
abbreviations, and — if relevant languages were provided — equivalent terms \
in those languages (e.g. German "Werkstudent" / "Praktikant" for "intern" / \
"working student").

Output ONLY a JSON object mapping each original title to a list of \
5-8 search-friendly variants. No commentary, no markdown fences.

Example shape:
{
  "AI Engineering Intern": ["AI Intern", "Machine Learning Intern", "Werkstudent KI", ...]
}
"""


def expand_titles(
    target_titles: list[str],
    languages: list[str],
    client: Groq,
) -> list[str]:
    """Return a flat, deduped list of expanded search terms (originals included)."""

    if not target_titles:
        return []

    user_msg = (
        f"Titles to expand: {json.dumps(target_titles)}\n"
        f"Candidate languages: {json.dumps(languages)}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        reasoning_format="hidden",
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        expansion_map = json.loads(raw)
    except json.JSONDecodeError:
        # Non-fatal — fall back to just using the original titles
        return list(dict.fromkeys(target_titles))

    all_terms = list(target_titles)
    for variants in expansion_map.values():
        all_terms.extend(variants)

    # dedupe, case-insensitive, preserve order
    seen = set()
    deduped = []
    for term in all_terms:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(term.strip())

    return deduped
