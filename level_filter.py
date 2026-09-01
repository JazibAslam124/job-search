"""Step 5.5 of the pipeline: hard filter listings down to intern /
working-student level roles, restricted to Bavaria (onsite) or remote
anywhere in Germany — before spending LLM calls ranking anything off-target.

Plain keyword matching, no LLM. This runs on both sources uniformly, since
JSearch's employment_types filter and Adzuna's API don't reliably agree on
what counts as "internship" across postings.
"""

from models import JobListing

# Titles containing any of these are almost never intern/working-student roles.
SENIORITY_EXCLUDE = [
    "senior", "sr.", " sr ", "lead ", "principal", "staff ", "head of",
    "director", "manager", "chef ", "architect", "vp ", "vice president",
    "expert ", "specialist ",
]

# Require at least one of these in the title OR description to count as
# intern/working-student level.
LEVEL_INCLUDE = [
    "intern", "internship", "praktik", "werkstudent", "working student",
    "trainee", "duales studium", "dual student", "student assistant",
    "hiwi", "co-op", "coop",
]

BAVARIA_KEYWORDS = [
    "bavaria", "bayern", "munich", "münchen", "muenchen", "nuremberg",
    "nürnberg", "nuernberg", "augsburg", "regensburg", "würzburg",
    "wuerzburg", "wurzburg", "ingolstadt", "fürth", "fuerth", "furth",
    "erlangen", "bamberg", "bayreuth", "passau", "deggendorf", "landshut",
    "aschaffenburg", "rosenheim", "kempten", "schweinfurt", "amberg",
    "straubing", "weiden", "hof", "freising", "dachau", "fürstenfeldbruck",
]

REMOTE_KEYWORDS = ["remote", "home office", "homeoffice", "home-office", "work from home", "wfh"]


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = (text or "").lower()
    return any(kw in text for kw in keywords)


def filter_listings(listings: list[JobListing], debug: bool = True) -> list[JobListing]:
    """Return only listings that look like intern/working-student roles
    located in Bavaria (onsite) or remote anywhere in Germany.

    debug=True prints the reason each dropped listing was excluded, so
    keyword mismatches (unexpected title/location phrasing) are visible
    instead of just silently vanishing.
    """
    kept = []
    for job in listings:
        title = job.title or ""
        desc = job.description or ""
        loc = job.location or ""

        if _contains_any(title, SENIORITY_EXCLUDE):
            if debug:
                print(f"[filter] DROP (seniority) '{title}' | loc='{loc}'")
            continue

        if not (_contains_any(title, LEVEL_INCLUDE) or _contains_any(desc, LEVEL_INCLUDE)):
            if debug:
                print(f"[filter] DROP (not intern/wstudent) '{title}' | loc='{loc}'")
            continue

        is_remote = _contains_any(loc, REMOTE_KEYWORDS) or _contains_any(desc, REMOTE_KEYWORDS)
        is_bavaria = _contains_any(loc, BAVARIA_KEYWORDS)

        if not (is_remote or is_bavaria):
            if debug:
                print(f"[filter] DROP (location) '{title}' | loc='{loc}'")
            continue

        kept.append(job)

    return kept
