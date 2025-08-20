# import duckdb
# import io
import os
# import pickle
# import pandas as pd
# from pathlib import Path
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# from typing import Dict, List, Tuple

from download_mimic_data import connect_mimic
from cohort_constractions import extract_raw

def main():
    """Main function to run data extraction locally with Google Drive dataset."""
    con = connect_mimic()

    initial_cohort_csv = "csvs/initial_cohort.csv"
    labs_csv = "csvs/labs_metadata.csv"
    vitals_csv = "csvs/vital_metadata.csv"

    print("Starting raw data extraction...")
    results = extract_raw(con, initial_cohort_csv, labs_csv, vitals_csv)

    print("\nExtraction completed successfully!")
    print(f"Cohort size: {len(results['cohort'])} patients")
    print(f"Lab events: {len(results['labs'])} records")
    print(f"Vital events: {len(results['vitals'])} records")
    print(f"Target labels: {len(results['targets'])} patients")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    cohort_path = os.path.join(output_dir, "cohort.csv")
    labs_path = os.path.join(output_dir, "labs.csv")
    vitals_path = os.path.join(output_dir, "vitals.csv")
    targets_path = os.path.join(output_dir, "targets.csv")

    results['cohort'].to_csv(cohort_path, index=False)
    results['labs'].to_csv(labs_path, index=False)
    results['vitals'].to_csv(vitals_path, index=False)
    results['targets'].to_csv(targets_path, index=False)

    print(f"\nResults saved to CSV files:")
    print(f"  - Cohort data: {cohort_path} ({len(results['cohort'])} rows)")
    print(f"  - Labs data: {labs_path} ({len(results['labs'])} rows)")
    print(f"  - Vitals data: {vitals_path} ({len(results['vitals'])} rows)")
    print(f"  - Targets data: {targets_path} ({len(results['targets'])} rows)")

    print("\n--- Cohort Sample ---")
    print(results['cohort'].head())

    print("\n--- Labs Sample ---")
    print(results['labs'].head())

    print("\n--- Vitals Sample ---")
    print(results['vitals'].head())

    print("\n--- Targets Sample ---")
    print(results['targets'].head())

    return results


if __name__ == "__main__":
    main()