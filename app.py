"""Streamlit UI for the job search agent.

Wraps the exact same pipeline as main.py (cv_parser -> criteria_extractor ->
keyword_expander -> query_builder -> sources -> aggregator -> level_filter
-> ranker) behind a browser UI, so nothing about the underlying logic is
duplicated or re-implemented here.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from cv_parser import extract_cv_text
from criteria_extractor import extract_criteria
from keyword_expander import expand_titles
from query_builder import build_query_plans
from aggregator import merge_and_dedupe
from level_filter import filter_listings
from ranker import rank_listings
from sources import adzuna, jsearch

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Check Streamlit Cloud's secrets manager first, then fall back to a
    local .env / environment variable. This lets the same app.py work both
    locally (via .env) and once deployed to Streamlit Cloud (via st.secrets),
    without maintaining two separate config paths.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # no secrets.toml configured locally — that's fine
    return os.getenv(key, default)


st.set_page_config(page_title="Job Search Agent", page_icon="🔎", layout="wide")
st.title("🔎 Job Search Agent")
st.caption("Bavaria onsite + remote-in-Germany, internship / working-student roles only")

# --- Sidebar: config ---
with st.sidebar:
    st.header("Settings")

    groq_key = st.text_input(
        "GROQ_API_KEY",
        value=_get_secret("GROQ_API_KEY"),
        type="password",
        help="Falls back to Streamlit secrets / your .env value if left as-is.",
    )
    adzuna_id = st.text_input("ADZUNA_APP_ID", value=_get_secret("ADZUNA_APP_ID"))
    adzuna_key = st.text_input(
        "ADZUNA_APP_KEY", value=_get_secret("ADZUNA_APP_KEY"), type="password"
    )
    rapidapi_key = st.text_input(
        "RAPIDAPI_KEY", value=_get_secret("RAPIDAPI_KEY"), type="password"
    )
    adzuna_country = st.text_input(
        "Country code", value=_get_secret("ADZUNA_COUNTRY", "de")
    )

    st.divider()
    max_results = st.slider("Max results to show", 5, 50, 15)
    location_override = st.text_input(
        "Location override (optional)",
        placeholder="e.g. Bavaria, Germany",
    )
    remote_ok = st.checkbox("Include remote roles", value=True)

# --- Main: CV upload + run ---
uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
run_clicked = st.button("Search jobs", type="primary", disabled=uploaded_cv is None)

if run_clicked:
    if not groq_key:
        st.error("GROQ_API_KEY is required. Fill it in the sidebar or your .env.")
        st.stop()
    if not (adzuna_id and adzuna_key) and not rapidapi_key:
        st.error("Configure at least one job source (Adzuna or RapidAPI/JSearch) in the sidebar.")
        st.stop()

    client = Groq(api_key=groq_key)

    # Save the uploaded PDF to a temp path since cv_parser expects a filepath
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_cv.read())
        tmp_path = tmp.name

    status = st.status("Running pipeline...", expanded=True)

    try:
        status.write("📄 Reading CV...")
        cv_text = extract_cv_text(tmp_path)

        status.write("🧠 Extracting search criteria (LLM)...")
        criteria = extract_criteria(cv_text, client)
        if location_override:
            criteria.location = [location_override]
        criteria.remote_ok = remote_ok
        status.write(f"Target titles: {', '.join(criteria.target_titles)}")
        status.write(f"Location: {', '.join(criteria.location)}")

        status.write("🔤 Expanding search terms (LLM)...")
        expanded_titles = expand_titles(criteria.target_titles, criteria.languages, client)
        status.write(f"{len(expanded_titles)} search terms generated")

        status.write("🗺️ Building queries...")
        plans = build_query_plans(criteria, expanded_titles)
        status.write(f"{len(plans)} queries planned")

        status.write("🌐 Searching job sources...")
        all_listings = []
        progress = st.progress(0.0)
        for i, plan in enumerate(plans):
            if adzuna_id and adzuna_key:
                all_listings.extend(
                    adzuna.search(plan, adzuna_id, adzuna_key, country=adzuna_country)
                )
            if rapidapi_key:
                all_listings.extend(
                    jsearch.search(plan, rapidapi_key, country=adzuna_country)
                )
            progress.progress((i + 1) / len(plans))

        deduped = merge_and_dedupe(all_listings)
        status.write(f"{len(all_listings)} raw listings -> {len(deduped)} after dedupe")

        deduped = filter_listings(deduped, debug=False)
        status.write(f"{len(deduped)} remain after filtering to intern/working-student + Bavaria/remote")

        if not deduped:
            status.update(label="No listings found", state="error")
            st.warning(
                "No listings matched. Try widening the location, or double-check your API keys/quotas."
            )
            st.stop()

        status.write("⭐ Scoring fit against your CV (LLM)...")
        ranked = rank_listings(deduped, cv_text, client)
        top_results = ranked[:max_results]

        status.update(label="Done!", state="complete")

    finally:
        os.unlink(tmp_path)

    st.subheader(f"Top {len(top_results)} matches")
    for i, job in enumerate(top_results, 1):
        score = job.fit_score if job.fit_score is not None else "?"
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{i}. {job.title}** — {job.company}")
                st.caption(f"{job.location} · via {job.source}")
                if job.fit_reason:
                    st.write(job.fit_reason)
                st.link_button("Open listing", job.url)
            with col2:
                st.metric("Fit", f"{score}%" if score != "?" else "?")

    # Downloadable results
    import csv
    import io

    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=[
            "fit_score", "title", "company", "location",
            "fit_reason", "url", "source", "posted_date", "salary",
        ],
    )
    writer.writeheader()
    for j in top_results:
        writer.writerow(j.to_dict())

    st.download_button(
        "Download results as CSV",
        data=csv_buf.getvalue(),
        file_name="results.csv",
        mime="text/csv",
    )
