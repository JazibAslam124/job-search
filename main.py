"""Orchestrates the full pipeline:

  CV --> extract criteria --> expand keywords --> build queries
      --> fetch from all sources --> dedupe --> rank against CV --> output

Usage:
    python main.py --cv resume.pdf
    python main.py --cv resume.pdf --location "Munich, Germany" --max-results 20
"""

import argparse
import csv
import json
import os
import sys

from groq import Groq
from dotenv import load_dotenv

from cv_parser import extract_cv_text
from criteria_extractor import extract_criteria
from keyword_expander import expand_titles
from query_builder import build_query_plans
from aggregator import merge_and_dedupe
from level_filter import filter_listings
from ranker import rank_listings
from sources import adzuna, jsearch


def parse_args():
    p = argparse.ArgumentParser(description="AI internship/job search agent")
    p.add_argument("--cv", required=True, help="Path to CV PDF")
    p.add_argument(
        "--location",
        default=None,
        help="Override location to search (defaults to what's inferred from the CV)",
    )
    p.add_argument(
        "--remote-ok",
        action="store_true",
        help="Bias search toward remote-friendly roles",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=15,
        help="Max number of ranked results to display/save (default: 15)",
    )
    p.add_argument(
        "--out",
        default="results",
        help="Output filename prefix for results.json / results.csv (default: results)",
    )
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        sys.exit("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and fill it in.")

    adzuna_id = os.getenv("ADZUNA_APP_ID")
    adzuna_key = os.getenv("ADZUNA_APP_KEY")
    adzuna_country = os.getenv("ADZUNA_COUNTRY", "de")
    rapidapi_key = os.getenv("RAPIDAPI_KEY")

    if not adzuna_id and not rapidapi_key:
        sys.exit(
            "ERROR: No job source configured. Set at least ADZUNA_APP_ID/"
            "ADZUNA_APP_KEY or RAPIDAPI_KEY in .env."
        )

    client = Groq(api_key=groq_key)

    # --- Step 1: parse CV ---
    print(f"[1/6] Reading CV: {args.cv}")
    cv_text = extract_cv_text(args.cv)

    # --- Step 2: extract structured criteria ---
    print("[2/6] Extracting search criteria from CV (LLM)...")
    criteria = extract_criteria(cv_text, client)
    if args.location:
        criteria.location = [args.location]
    if args.remote_ok:
        criteria.remote_ok = True
    print(f"       target_titles: {criteria.target_titles}")
    print(f"       location: {criteria.location}")

    # --- Step 3: expand keywords ---
    print("[3/6] Expanding titles for search coverage (LLM)...")
    expanded_titles = expand_titles(criteria.target_titles, criteria.languages, client)
    print(f"       {len(expanded_titles)} search terms generated")

    # --- Step 4: build queries (deterministic) ---
    print("[4/6] Building search queries...")
    plans = build_query_plans(criteria, expanded_titles)
    print(f"       {len(plans)} queries planned")

    # --- Step 5: fetch from all sources + dedupe ---
    print("[5/6] Searching job sources...")
    all_listings = []
    for plan in plans:
        query_count = 0
        if adzuna_id and adzuna_key:
            results = adzuna.search(plan, adzuna_id, adzuna_key, country=adzuna_country)
            query_count += len(results)
            all_listings.extend(results)
        if rapidapi_key:
            results = jsearch.search(plan, rapidapi_key, country=adzuna_country)
            query_count += len(results)
            all_listings.extend(results)
        print(f"       '{plan.keywords}' in '{plan.location}' -> {query_count} results")

    deduped = merge_and_dedupe(all_listings)
    print(f"       {len(all_listings)} raw listings -> {len(deduped)} after dedupe")

    deduped = filter_listings(deduped)
    print(f"       {len(deduped)} remain after filtering to intern/working-student + Bavaria/remote")

    if not deduped:
        print("\nNo listings found. Try broadening --location or check your API keys.")
        return

    # --- Step 6: rank against CV ---
    print("[6/6] Scoring fit against your CV (LLM)...")
    ranked = rank_listings(deduped, cv_text, client)
    top_results = ranked[: args.max_results]

    # --- Output ---
    print(f"\n{'='*70}\nTOP {len(top_results)} MATCHES\n{'='*70}")
    for i, job in enumerate(top_results, 1):
        score = job.fit_score if job.fit_score is not None else "?"
        print(f"\n{i}. [{score}%] {job.title} — {job.company} ({job.location})")
        print(f"   {job.fit_reason or ''}")
        print(f"   {job.url}")

    _save_json(top_results, f"{args.out}.json")
    _save_csv(top_results, f"{args.out}.csv")
    print(f"\nSaved: {args.out}.json, {args.out}.csv")


def _save_json(listings, path):
    with open(path, "w") as f:
        json.dump([j.to_dict() for j in listings], f, indent=2)


def _save_csv(listings, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fit_score", "title", "company", "location",
                "fit_reason", "url", "source", "posted_date", "salary",
            ],
        )
        writer.writeheader()
        for j in listings:
            writer.writerow(j.to_dict())


if __name__ == "__main__":
    main()
