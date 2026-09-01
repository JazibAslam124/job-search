"""Step 4 of the pipeline: deterministic (no LLM) construction of API query
parameters for each source, from structured criteria + expanded keywords.

Keeping this deterministic means query construction is predictable,
debuggable, and doesn't cost an LLM call per query.
"""

from dataclasses import dataclass

from models import SearchCriteria


@dataclass
class QueryPlan:
    """One search to run, generic across sources."""

    keywords: str
    location: str


def build_query_plans(
    criteria: SearchCriteria,
    expanded_titles: list[str],
    max_queries: int = 18,
) -> list[QueryPlan]:
    """Turn criteria + expanded titles into a bounded list of concrete
    queries to run against each source.

    Onsite search targets Bavaria as one broad region (job APIs geocode
    "Bavaria" across the whole state) rather than querying each city
    separately, which would waste query budget on redundant coverage of
    the same region. Remote search covers all of Germany, since remote
    roles aren't location-bound.
    """

    titles = expanded_titles or criteria.target_titles or ["internship"]

    search_targets = ["Bavaria, Germany"]
    if criteria.remote_ok:
        search_targets.append("Germany")

    titles_per_target = max(1, max_queries // len(search_targets))
    chosen_titles = titles[:titles_per_target]

    plans = [
        QueryPlan(keywords=title, location="Bavaria, Germany")
        for title in chosen_titles
    ]

    if criteria.remote_ok:
        for title in chosen_titles:
            # Embed "remote" in the keyword text itself since neither
            # Adzuna nor JSearch's keyword search takes a clean remote-only
            # filter param that works reliably for Germany.
            plans.append(QueryPlan(keywords=f"{title} remote", location="Germany"))

    return plans[:max_queries]
