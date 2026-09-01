"""Adzuna job search API client.

Free tier signup: https://developer.adzuna.com/
Docs: https://developer.adzuna.com/docs/search
"""

import requests

from models import JobListing
from query_builder import QueryPlan

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


def search(
    plan: QueryPlan,
    app_id: str,
    app_key: str,
    country: str = "de",
    results_per_query: int = 10,
) -> list[JobListing]:
    """Run one query against Adzuna and return normalized listings.

    Returns an empty list (rather than raising) on API errors, so one bad
    query doesn't kill the whole pipeline — errors are printed to stderr.
    """

    if not app_id or not app_key:
        return []

    url = BASE_URL.format(country=country)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_query,
        "what": plan.keywords,
        "where": plan.location,
        "content-type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[adzuna] request failed for '{plan.keywords}': {e}")
        return []
    except ValueError as e:
        print(f"[adzuna] bad JSON response for '{plan.keywords}': {e}")
        return []

    listings = []
    for item in data.get("results", []):
        listings.append(
            JobListing(
                title=item.get("title", "").strip(),
                company=(item.get("company") or {}).get("display_name", "Unknown"),
                location=(item.get("location") or {}).get("display_name", ""),
                url=item.get("redirect_url", ""),
                source="adzuna",
                description=(item.get("description") or "")[:800],
                posted_date=item.get("created"),
                salary=_format_salary(item),
            )
        )

    return listings


def _format_salary(item: dict) -> str | None:
    lo = item.get("salary_min")
    hi = item.get("salary_max")
    if lo and hi:
        return f"{int(lo):,} - {int(hi):,}"
    if lo:
        return f"from {int(lo):,}"
    return None
