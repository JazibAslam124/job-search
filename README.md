# AI Job/Internship Search Agent

Finds internship / working-student roles that match your CV, using a hybrid
pipeline: deterministic API search + LLM reasoning for extraction, keyword
expansion, and fit-ranking.

## Pipeline

```
CV (PDF) --> [LLM] Extract structured criteria
          --> [LLM] Expand titles/keywords for coverage
          --> [code] Build API queries (Adzuna, JSearch)
          --> [code] Fetch + dedupe listings
          --> [LLM] Rank & filter listings against your CV
          --> Ranked results (console + JSON + CSV)
```

Only steps involving *judgment* (reading the CV, expanding synonyms, judging
fit) use the LLM. Searching, fetching, and deduping are plain deterministic
code — cheap, fast, and debuggable.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

You need:

- **ANTHROPIC_API_KEY** — for the LLM reasoning steps.
  Get one at https://console.anthropic.com/
- **ADZUNA_APP_ID** / **ADZUNA_APP_KEY** — free tier at
  https://developer.adzuna.com/
- **RAPIDAPI_KEY** — for JSearch, free tier at
  https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
  (optional — the agent still works with only Adzuna configured)

## 2. Run

```bash
python main.py --cv /path/to/your_cv.pdf
```

Optional flags:

```bash
python main.py --cv resume.pdf \
    --location "Bavaria, Germany" \
    --remote-ok \
    --max-results 20 \
    --out results
```

This produces:
- Ranked results printed to the console
- `results.json` — full structured output
- `results.csv` — spreadsheet-friendly version

## 3. Re-running automatically (optional)

To check for new postings daily, add a cron job (Linux/Mac):

```bash
crontab -e
# run every morning at 8am
0 8 * * * cd /path/to/job_agent && ./venv/bin/python main.py --cv resume.pdf --out results >> agent.log 2>&1
```

## Project layout

```
job_agent/
├── main.py                  # orchestrator — runs the full pipeline
├── cv_parser.py              # extracts raw text from the CV PDF
├── criteria_extractor.py     # LLM: CV text -> structured JSON criteria
├── keyword_expander.py       # LLM: title expansion for search coverage
├── query_builder.py          # deterministic: criteria -> API query params
├── aggregator.py              # dedupe + merge listings from all sources
├── ranker.py                  # LLM: score/filter listings against the CV
├── sources/
│   ├── adzuna.py               # Adzuna API client
│   └── jsearch.py              # JSearch (RapidAPI) client
├── models.py                   # shared dataclasses
├── requirements.txt
├── .env.example
└── README.md
```

## Notes / things to know

- **No LinkedIn scraping.** LinkedIn actively blocks and bans scraping/automation
  and it violates their ToS. This project only uses official APIs
  (Adzuna, JSearch — which itself aggregates Google for Jobs results,
  including many LinkedIn/Indeed postings legally re-surfaced).
- **Add more sources easily**: drop a new file in `sources/` following the
  same interface (`search(query_params) -> list[JobListing]`) and register
  it in `main.py`.
- **Applying to jobs is NOT automated** — this agent finds and ranks, it does
  not fill out or submit applications. That's a separate, much riskier
  feature (every ATS form is different) that's intentionally out of scope here.
- If you only configure Adzuna (skip JSearch), the agent still runs fine with
  one source — it just has less coverage.
