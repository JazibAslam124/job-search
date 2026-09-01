"""Shared data structures used across the pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JobListing:
    """A normalized job listing, regardless of which source it came from."""

    title: str
    company: str
    location: str
    url: str
    source: str
    description: str = ""
    posted_date: Optional[str] = None
    salary: Optional[str] = None

    # filled in later by the ranker
    fit_score: Optional[int] = None
    fit_reason: Optional[str] = None

    def dedupe_key(self) -> str:
        """Rough key for deduping the same job posted on multiple boards."""
        return f"{self.title.strip().lower()}|{self.company.strip().lower()}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "source": self.source,
            "posted_date": self.posted_date,
            "salary": self.salary,
            "fit_score": self.fit_score,
            "fit_reason": self.fit_reason,
        }


@dataclass
class SearchCriteria:
    """Structured criteria extracted from the CV (LLM output, step 2)."""

    target_titles: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    seniority: str = "intern"
    education_status: str = ""
    location: list[str] = field(default_factory=list)
    remote_ok: bool = True
    languages: list[str] = field(default_factory=list)
    industries_of_interest: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchCriteria":
        return cls(
            target_titles=d.get("target_titles", []),
            skills=d.get("skills", []),
            seniority=d.get("seniority", "intern"),
            education_status=d.get("education_status", ""),
            location=d.get("location", []),
            remote_ok=d.get("remote_ok", True),
            languages=d.get("languages", []),
            industries_of_interest=d.get("industries_of_interest", []),
            exclude=d.get("exclude", []),
        )
