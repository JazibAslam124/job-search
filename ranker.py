"""Step 6 of the pipeline: LLM scores each listing's fit against the CV.

This is the highest-value reasoning step — it catches things keyword
matching can't, e.g. a title match that actually requires 5 years of Java
experience, or a strong match that's phrased differently than the search
terms.

Listings are batched to keep prompts a manageable size.
"""

import json

from groq import Groq

from models import JobListing

MODEL = "qwen/qwen3.6-27b"
BATCH_SIZE = 12

SYSTEM_PROMPT = """You are scoring job listings for fit against a candidate's \
CV. For each listing, return a fit_score from 0-100 and a one-sentence \
fit_reason explaining the score in plain language, referencing specific \
overlaps or gaps (skills, seniority, language requirements, etc).

Score generously for potential/transferable fit (this is for internships \
and entry-level roles — candidates are not expected to already have \
professional experience in every listed skill). Score low (below 30) if the \
listing requires years of professional experience clearly beyond an \
intern/working-student level, or requires a language/skill the candidate \
clearly lacks and the listing states it as required.

Output ONLY a JSON array, one object per listing, in the SAME ORDER as the \
listings were given, with this shape:
[{"fit_score": int, "fit_reason": string}, ...]
No commentary, no markdown fences, no extra fields.
"""


def rank_listings(
    listings: list[JobListing],
    cv_text: str,
    client: Groq,
) -> list[JobListing]:
    """Mutates and returns listings with fit_score / fit_reason populated,
    sorted by fit_score descending.
    """

    for start in range(0, len(listings), BATCH_SIZE):
        batch = listings[start : start + BATCH_SIZE]
        _score_batch(batch, cv_text, client)

    listings.sort(key=lambda j: (j.fit_score or 0), reverse=True)
    return listings


def _score_batch(
    batch: list[JobListing],
    cv_text: str,
    client: Groq,
) -> None:
    listing_summaries = [
        {
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "description_excerpt": j.description[:500],
        }
        for j in batch
    ]

    user_msg = (
        f"CANDIDATE CV:\n{cv_text}\n\n"
        f"LISTINGS TO SCORE ({len(batch)} total):\n"
        f"{json.dumps(listing_summaries, indent=2)}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
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
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Non-fatal: leave this batch unscored (fit_score stays None,
        # sorts to the bottom) rather than crashing the whole run.
        print(f"[ranker] failed to parse scores for a batch of {len(batch)}")
        return

    for job, score in zip(batch, scores):
        job.fit_score = score.get("fit_score")
        job.fit_reason = score.get("fit_reason")
