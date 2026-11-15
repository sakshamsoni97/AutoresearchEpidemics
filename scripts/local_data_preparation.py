"""Prepare US county-level COVID-19 case and death files for training.

Converted from the first five cells of local_data_preparation.ipynb.
"""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "covid19_us"
NAME_PREFIX = "Cases_US_"
MAX_DATE = "2021-01-01"


def process_file(f):
    """Write the 90-day training window for one raw file.

    Returns True if an output file was written, False if the file was skipped.
    """
    raw_data = pd.read_csv(RAW_DATA_DIR / f)
    raw_data = raw_data[raw_data.date < MAX_DATE]
    if len(raw_data[raw_data.day_since100 > 0]) < 90:
        print(f"Skipped {f}: fewer than 90 days with day_since100 > 0")
        return False

    file_suffix = re.match(r"Cases_(\w+).csv", f).group(1)
    data = raw_data[["country", "province", "date", "case_cnt", "death_cnt"]].copy()
    data.date = pd.to_datetime(data.date)
    start_date = datetime.strptime(
        raw_data[raw_data.day_since100 == 0].iloc[0].date, "%Y-%m-%d"
    ).date()
    end_date = start_date + timedelta(days=90)
    data = data[(data.date >= str(start_date)) & (data.date <= str(end_date))]

    output_path = OUTPUT_DIR / f"cases_deaths_{file_suffix}.csv"
    data.to_csv(output_path, index=False)
    print(f"Success: wrote {output_path.name} ({len(data)} rows)")
    return True


def process_population():
    """Write the US rows of Population_Global.csv to Population_US_States.csv."""
    pop_global = pd.read_csv(RAW_DATA_DIR / "Population_Global.csv")
    pop_us = pop_global[pop_global.Country == "US"].copy()

    output_path = PROJECT_ROOT / "data" / "training" / "Population_US_States.csv"
    pop_us.to_csv(output_path, index=False)
    print(f"Success: wrote {output_path.name} ({len(pop_us)} rows)")


def main():
    """Run the preparation step. Returns a process exit code."""
    if not RAW_DATA_DIR.is_dir():
        print(f"Error: raw data directory not found: {RAW_DATA_DIR}", file=sys.stderr)
        return 1
    if not OUTPUT_DIR.is_dir():
        print(f"Error: output directory not found: {OUTPUT_DIR}", file=sys.stderr)
        return 1

    try:
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith(NAME_PREFIX)]
    except OSError as e:
        print(f"Error: could not list {RAW_DATA_DIR}: {e}", file=sys.stderr)
        return 1

    failed = 0
    try:
        process_population()
    except (OSError, ValueError, KeyError, AttributeError) as e:
        failed += 1
        print(f"Error processing Population_Global.csv: {type(e).__name__}: {e}", file=sys.stderr)

    if not raw_files:
        print(f"No files starting with '{NAME_PREFIX}' found in {RAW_DATA_DIR}")

    written, skipped = 0, 0
    for f in raw_files:
        try:
            if process_file(f):
                written += 1
            else:
                skipped += 1
        except (OSError, ValueError, KeyError, IndexError, AttributeError) as e:
            # One bad file should not stop the rest of the batch.
            failed += 1
            print(f"Error processing {f}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"Done: {written} written, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
