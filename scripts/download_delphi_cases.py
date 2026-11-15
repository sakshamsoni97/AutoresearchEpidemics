"""Download Cases_US_*.csv and Population_Global.csv from COVIDAnalytics/DELPHI data_sandbox/processed.

Usage:
    python scripts/download_delphi_cases.py

Files are saved to data/raw_data/ relative to the project root.
Uses only the Python standard library.
"""

import json
import sys
import urllib.request
from pathlib import Path

REPO = "COVIDAnalytics/DELPHI"
REMOTE_DIR = "data_sandbox/processed"
BRANCH = "master"
NAME_PREFIX = "Cases_US_"
EXTENSION = ".csv"
EXTRA_FILES = {"Population_Global.csv"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw_data"


def list_remote_files() -> list[dict]:
    """Return the GitHub API content entries for the remote folder."""
    url = f"https://api.github.com/repos/{REPO}/contents/{REMOTE_DIR}?ref={BRANCH}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        entries = json.load(resp)
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected response listing {REMOTE_DIR}: {entries!r}")
    return entries


def download(url: str, dest: Path) -> None:
    """Fetch url and write it to dest atomically."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.replace(dest)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = [
        e for e in list_remote_files()
        if e.get("type") == "file"
        and (
            (e["name"].startswith(NAME_PREFIX) and e["name"].endswith(EXTENSION))
            or e["name"] in EXTRA_FILES
        )
    ]
    missing = EXTRA_FILES - {e["name"] for e in matches}
    if missing:
        print(f"Warning: not found in {REMOTE_DIR}: {', '.join(sorted(missing))}")
    if not matches:
        print(f"No files matching {NAME_PREFIX}*{EXTENSION} or {', '.join(sorted(EXTRA_FILES))} in {REMOTE_DIR}")
        return 1

    print(f"Found {len(matches)} matching files; saving to {OUT_DIR}")
    failures = []
    for entry in sorted(matches, key=lambda e: e["name"]):
        dest = OUT_DIR / entry["name"]
        try:
            download(entry["download_url"], dest)
            print(f"  ok   {entry['name']}")
        except Exception as exc:  # keep going; report at the end
            failures.append((entry["name"], exc))
            print(f"  FAIL {entry['name']}: {exc}")

    print(f"Done: {len(matches) - len(failures)} downloaded, {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
