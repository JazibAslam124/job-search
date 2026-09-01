"""Step 5 of the pipeline: merge listings from all sources and dedupe.

Plain code, no LLM — deduping by (title, company) is cheap and reliable
enough for this use case.
"""

from models import JobListing


def merge_and_dedupe(all_listings: list[JobListing]) -> list[JobListing]:
    seen = {}
    for listing in all_listings:
        key = listing.dedupe_key()
        if key not in seen:
            seen[key] = listing
    return list(seen.values())
