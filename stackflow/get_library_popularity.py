import os
import time
import math
import requests
import pandas as pd


LIBRARIES = [
    {
        "library": "stdlib",
        "stackoverflow_tag": None,  
        "stackoverflow_query": "python logging module",
        "github_repo": None,       
        "notes": "Python standard library logging"
    },
    {
        "library": "loguru",
        "stackoverflow_tag": "loguru",
        "stackoverflow_query": "loguru python",
        "github_repo": "Delgan/loguru",
        "notes": ""
    },
    {
        "library": "structlog",
        "stackoverflow_tag": "structlog",
        "stackoverflow_query": "structlog python",
        "github_repo": "hynek/structlog",
        "notes": ""
    },
    {
        "library": "logbook",
        "stackoverflow_tag": "logbook",
        "stackoverflow_query": "logbook python logging",
        "github_repo": "getlogbook/logbook",
        "notes": ""
    },
    {
        "library": "pythonjsonlogger",
        "stackoverflow_tag": None,
        "stackoverflow_query": "python-json-logger OR pythonjsonlogger",
        "github_repo": "madzak/python-json-logger",
        "notes": "Query-based SO lookup only"
    },
    {
        "library": "picologging",
        "stackoverflow_tag": None,
        "stackoverflow_query": "picologging python",
        "github_repo": "microsoft/picologging",
        "notes": "Query-based SO lookup only"
    },
    {
        "library": "aiologger",
        "stackoverflow_tag": "aiologger",
        "stackoverflow_query": "aiologger python",
        "github_repo": "async-worker/aiologger",
        "notes": ""
    },
]

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

STACKEXCHANGE_BASE = "https://api.stackexchange.com/2.3"
GITHUB_BASE = "https://api.github.com"


def github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "logging-library-popularity-script"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def get_stackoverflow_tag_count(tag: str) -> int | None:
    """Return Stack Overflow tag count if the tag exists, else None."""
    if not tag:
        return None

    url = f"{STACKEXCHANGE_BASE}/tags/{tag}/info"
    params = {"site": "stackoverflow"}

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            return None
        return int(items[0].get("count", 0))
    except Exception:
        return None


def get_stackoverflow_query_count(query: str) -> int | None:
    url = f"{STACKEXCHANGE_BASE}/search/advanced"
    params = {
        "site": "stackoverflow",
        "q": query,
        "pagesize": 100,
        "order": "desc",
        "sort": "relevance",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "total" in data:
            return int(data["total"])

        return int(len(data.get("items", [])))
    except Exception:
        return None


def get_github_repo_stats(repo: str) -> dict:
    empty = {
        "github_stars": None,
        "github_forks": None,
        "github_open_issues": None,
        "github_watchers": None,
        "github_archived": None,
        "github_created_at": None,
        "github_updated_at": None,
    }

    if not repo:
        return empty

    url = f"{GITHUB_BASE}/repos/{repo}"

    try:
        r = requests.get(url, headers=github_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()

        return {
            "github_stars": data.get("stargazers_count"),
            "github_forks": data.get("forks_count"),
            "github_open_issues": data.get("open_issues_count"),
            "github_watchers": data.get("subscribers_count"),
            "github_archived": data.get("archived"),
            "github_created_at": data.get("created_at"),
            "github_updated_at": data.get("updated_at"),
        }
    except Exception:
        return empty


def safe_log_score(x):
    if x is None or x <= 0:
        return 0.0
    return math.log10(x + 1)


def collect_library_data():
    rows = []

    for lib in LIBRARIES:
        name = lib["library"]
        print(f"Collecting data for: {name}")

        so_tag_count = get_stackoverflow_tag_count(lib["stackoverflow_tag"])
        time.sleep(0.2)

        so_query_count = get_stackoverflow_query_count(lib["stackoverflow_query"])
        time.sleep(0.2)

        gh = get_github_repo_stats(lib["github_repo"])
        time.sleep(0.2)

        row = {
            "library": name,
            "stackoverflow_tag": lib["stackoverflow_tag"],
            "stackoverflow_query": lib["stackoverflow_query"],
            "stackoverflow_tag_count": so_tag_count,
            "stackoverflow_query_count": so_query_count,
            "github_repo": lib["github_repo"],
            "notes": lib["notes"],
            **gh,
        }

        row["stackoverflow_primary_count"] = (
            so_tag_count if so_tag_count is not None else so_query_count
        )

        row["popularity_score"] = round(
            safe_log_score(row["stackoverflow_primary_count"])
            + safe_log_score(row["github_stars"])
            + 0.5 * safe_log_score(row["github_forks"]),
            4
        )

        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame):
    raw_csv = "python_logging_library_popularity_raw.csv"
    ranked_csv = "python_logging_library_popularity_ranked.csv"

    df.to_csv(raw_csv, index=False)

    ranked = df.sort_values(
        by=["popularity_score", "stackoverflow_primary_count", "github_stars"],
        ascending=False
    ).reset_index(drop=True)

    ranked["rank"] = ranked.index + 1
    ranked.to_csv(ranked_csv, index=False)

    print("\nSaved files:")
    print(f" - {raw_csv}")
    print(f" - {ranked_csv}")

    print("\nTop libraries by popularity score:")
    display_cols = [
        "rank",
        "library",
        "popularity_score",
        "stackoverflow_primary_count",
        "github_stars",
        "github_forks",
        "github_open_issues",
    ]
    print(ranked[display_cols].to_string(index=False))


if __name__ == "__main__":
    df = collect_library_data()
    save_outputs(df)