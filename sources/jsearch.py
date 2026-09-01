"""JSearch (via RapidAPI) job search client.

JSearch aggregates Google for Jobs results, which itself pulls from
LinkedIn, Indeed, Glassdoor and company career pages — giving broad
coverage through one legitimate API instead of scraping each site.

Free tier signup: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
"""

import requests

from models import JobListing
from query_builder import QueryPlan

BASE_URL = "https://jsearch.p.rapidapi.com/search-v2"


def search(
    plan: QueryPlan,
    rapidapi_key: str,
    results_per_query: int = 10,
    country: str = "de",
    language: str = "en",
) -> list[JobListing]:
    """Run one query against JSearch and return normalized listings.

    Uses the /search-v2 endpoint (JSearch retired the old /search path on
    RapidAPI in favor of this one). /search-v2 uses cursor-based pagination,
    but a cursor is only needed to fetch page 2+; the first page is returned
    without one, which is all this function asks for.

    Returns an empty list (rather than raising) on API errors, so one bad
    query doesn't kill the whole pipeline.
    """

    if not rapidapi_key:
        return []

    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{plan.keywords} in {plan.location}",
        "country": country,
        "language": language,
        "employment_types": "INTERN,PARTTIME",
    }

    try:
        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[jsearch] request failed for '{plan.keywords}': {e}")
        return []
    except ValueError as e:
        print(f"[jsearch] bad JSON response for '{plan.keywords}': {e}")
        return []

    # /search-v2 nests results under data.jobs, with a cursor for pagination.
    raw_data = data.get("data")
    if isinstance(raw_data, list):
        job_items = raw_data
    elif isinstance(raw_data, dict):
        if "jobs" in raw_data:
            job_items = raw_data["jobs"]  # correct key; may legitimately be empty
        else:
            job_items = raw_data.get("results") or raw_data.get("items") or []
            if not job_items:
                print(
                    f"[jsearch] unrecognized 'data' shape for '{plan.keywords}'. "
                    f"Keys found: {list(raw_data.keys())}"
                )
    else:
        job_items = []
        print(
            f"[jsearch] unexpected response shape for '{plan.keywords}'. "
            f"Top-level keys: {list(data.keys())}"
        )

    listings = []
    for item in job_items[:results_per_query]:
        listings.append(
            JobListing(
                title=item.get("job_title", "").strip(),
                company=item.get("employer_name", "Unknown"),
                location=_format_location(item),
                url=item.get("job_apply_link", ""),
                source="jsearch",
                description=(item.get("job_description") or "")[:800],
                posted_date=item.get("job_posted_at_datetime_utc"),
                salary=None,
            )
        )

    return listings


def _format_location(item: dict) -> str:
    city = item.get("job_city") or ""
    country = item.get("job_country") or ""
    if item.get("job_is_remote"):
        return "Remote" + (f" ({country})" if country else "")
    return ", ".join(p for p in [city, country] if p)
