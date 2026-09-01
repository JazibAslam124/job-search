"""Step 2 of the pipeline: LLM turns raw CV text into structured search
criteria (JSON). This is a constrained extraction task, not freeform
generation, which keeps it reliable.
"""

import json

from groq import Groq

from models import SearchCriteria

MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are a precise information-extraction engine. You read a \
CV/resume and output ONLY a single JSON object describing the candidate's job \
search criteria for internships / entry-level / working-student roles.

Rules:
- Output ONLY valid JSON. No markdown fences, no preamble, no commentary.
- Infer target job titles from the candidate's skills and projects, not just \
titles they've literally held before (they may have never held a paid role \
in their target field yet — that's normal for students).
- Include both English and, if the CV suggests a country where another \
language is spoken (e.g. Germany), that language's equivalent job-title terms \
(e.g. "Werkstudent", "Praktikum") in target_titles.
- seniority should reflect that this is an intern/working-student/entry-level \
search unless the CV clearly shows senior professional experience.
- education_status should note if they are a current student vs a graduate, \
since this affects which postings they qualify for.
- location should be a list of specific places/regions mentioned or implied \
(e.g. "open to relocation within Bavaria" -> list Bavaria + a few major \
Bavarian cities).
- languages should list spoken/written languages and proficiency as stated.
- exclude should list role types clearly mismatched to this candidate \
(e.g. if there's zero sales/marketing background, exclude "sales", \
"marketing" unless the CV suggests otherwise).

Required JSON shape:
{
  "target_titles": [string, ...],
  "skills": [string, ...],
  "seniority": string,
  "education_status": string,
  "location": [string, ...],
  "remote_ok": boolean,
  "languages": [string, ...],
  "industries_of_interest": [string, ...],
  "exclude": [string, ...]
}
"""


def extract_criteria(cv_text: str, client: Groq) -> SearchCriteria:
    """Call the LLM once to turn CV text into a SearchCriteria object."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        reasoning_format="hidden",
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CV TEXT:\n\n{cv_text}"},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()

    # Defensive cleanup in case the model wraps output in a code fence anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON for criteria extraction.\n"
            f"Raw output was:\n{raw}"
        ) from e

    return SearchCriteria.from_dict(data)
