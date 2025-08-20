import duckdb
import io
import os
import pickle
import pandas as pd
from pathlib import Path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from typing import Dict, List, Tuple

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def setup_google_drive_access():
    """Set up Google Drive API access with local authentication."""
    project_root = Path(__file__).parent.parent
    token_path = project_root / 'token.pickle'
    credentials_path = project_root / 'credentials.json'

    creds = None
    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)


def find_mimic_folder(drive_service):
    """Find the MIMIC-III folder in Google Drive."""
    results = drive_service.files().list(
        q=f"name='MIMIC-III' and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)"
    ).execute()

    folders = results.get('files', [])

    return folders[0]['id']


def list_files_in_folder(drive_service, folder_id):
    """List all files in a Google Drive folder for debugging."""
    print(f"Listing files in folder (ID: {folder_id}):")
    results = drive_service.files().list(
        q=f"parents in '{folder_id}'",
        fields="files(id, name, mimeType, size, modifiedTime)"
    ).execute()

    files = results.get('files', [])

    for file_info in files:
        name = file_info.get('name', 'Unknown')
        mime_type = file_info.get('mimeType', 'Unknown')
        size = file_info.get('size', 'Unknown')
        modified = file_info.get('modifiedTime', 'Unknown')
        file_id = file_info.get('id', 'Unknown')

        print(f"  - {name}")
        print(f"    MIME: {mime_type}")
        print(f"    Size: {size} bytes")
        print(f"    Modified: {modified}")
        print(f"    ID: {file_id}")
        print()

    return files


def download_file_from_drive(drive_service, file_name, folder_id, local_path):
    """Download a file from Google Drive to local path."""
    results = drive_service.files().list(
        q=f"name='{file_name}' and parents in '{folder_id}'",
        fields="files(id, name, mimeType, size)"
    ).execute()

    files = results.get('files', [])

    file_info = files[0]
    file_id = file_info['id']
    mime_type = file_info.get('mimeType', '')
    file_size = file_info.get('size', 'unknown')

    print(f"Found file: {file_name}")
    print(f"  - MIME type: {mime_type}")
    print(f"  - Size: {file_size} bytes")
    print(f"  - File ID: {file_id}")

    if mime_type == 'application/vnd.google-apps.shortcut':
        print("This is a Google Drive shortcut. Following the shortcut to get the actual file...")

        shortcut_details = drive_service.files().get(
            fileId=file_id, 
            fields='shortcutDetails'
        ).execute()

        target_id = shortcut_details['shortcutDetails']['targetId']
        print(f"Shortcut points to file ID: {target_id}")

        target_file = drive_service.files().get(
            fileId=target_id,
            fields='id, name, mimeType, size'
        ).execute()

        print(f"Target file details:")
        print(f"  - Name: {target_file.get('name', 'Unknown')}")
        print(f"  - MIME type: {target_file.get('mimeType', 'Unknown')}")
        print(f"  - Size: {target_file.get('size', 'Unknown')} bytes")

        file_id = target_id
        mime_type = target_file.get('mimeType', '')
        file_size = target_file.get('size', 'unknown')

        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        print("Starting download...")
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"Download progress: {progress}%")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())

        print(f"Successfully downloaded {file_name} to {local_path}")
        print(f"File size: {len(fh.getvalue())} bytes")

def download_file_from_drive_streaming(drive_service, file_name, folder_id, local_path):
    """
    Stream a large Drive file to disk in chunks (no RAM spike).
    - Finds file by name inside the known folder
    - Follows shortcuts
    - Writes 16MB chunks directly to local_path
    """
    # Locate the file in the folder
    resp = drive_service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents",
        fields="files(id, name, mimeType, size)",
        pageSize=10
    ).execute()
    files = resp.get('files', [])
    if not files:
        raise FileNotFoundError(f"'{file_name}' not found in folder {folder_id}")

    fi = files[0]
    file_id = fi['id']
    mime_type = fi.get('mimeType', '')
    remote_size = int(fi.get('size', '0') or 0)
    print(f"Found: {fi['name']} (size: {remote_size:,} bytes)")

    # Follow Google Drive shortcut
    if mime_type == 'application/vnd.google-apps.shortcut':
        sc = drive_service.files().get(fileId=file_id, fields='shortcutDetails').execute()
        file_id = sc['shortcutDetails']['targetId']
        tgt = drive_service.files().get(fileId=file_id, fields='id,name,size,mimeType').execute()
        remote_size = int(tgt.get('size', '0') or 0)
        print(f"Resolved shortcut → target: {tgt.get('name')} (size: {remote_size:,} bytes)")

    # Prepare request
    request = drive_service.files().get_media(fileId=file_id)

    # Ensure folder exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # If a partial file exists from a previous crash, start fresh (simplest + safest)
    if os.path.exists(local_path):
        os.remove(local_path)

    # Stream to disk in 16MB chunks
    with open(local_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%")

    # Verify size (optional)
    local_size = os.path.getsize(local_path)
    if remote_size and local_size != remote_size:
        raise IOError(f"Incomplete download: expected {remote_size:,} bytes, got {local_size:,} bytes")

    print(f"Saved to {local_path} ({local_size:,} bytes)")


def connect_mimic():
    """Main function to run data extraction locally with Google Drive dataset."""
    print("Setting up Google Drive access...")
    drive_service = setup_google_drive_access()
    print("Google Drive authentication successful")

    print("Finding MIMIC-III folder...")
    folder_id = find_mimic_folder(drive_service)
    print(f"Found MIMIC-III folder (ID: {folder_id})")

    print("\nListing all files in MIMIC-III folder:")
    list_files_in_folder(drive_service, folder_id)

    local_db_path = "data/mimiciii.duckdb"
    if not os.path.exists(local_db_path):
        print("Downloading MIMIC-III database...")
        download_file_from_drive_streaming(drive_service, "mimiciii.duckdb", folder_id, local_db_path)
        # download_file_from_drive(drive_service, "mimiciii.duckdb", folder_id, local_db_path)
    else:
        print("Using existing local database file")

    print("Connecting to MIMIC-III database...")
    con = duckdb.connect(local_db_path)
    print("Connected to database successfully")

    return con

    # initial_cohort_csv = "csvs/initial_cohort.csv"
    # labs_csv = "csvs/labs_metadata.csv"
    # vitals_csv = "csvs/vital_metadata.csv"

    # print("Starting raw data extraction...")
    # results = extract_raw(con, initial_cohort_csv, labs_csv, vitals_csv)

    # print("\nExtraction completed successfully!")
    # print(f"Cohort size: {len(results['cohort'])} patients")
    # print(f"Lab events: {len(results['labs'])} records")
    # print(f"Vital events: {len(results['vitals'])} records")
    # print(f"Target labels: {len(results['targets'])} patients")

    # output_dir = "data"
    # os.makedirs(output_dir, exist_ok=True)

    # cohort_path = os.path.join(output_dir, "cohort.csv")
    # labs_path = os.path.join(output_dir, "labs.csv")
    # vitals_path = os.path.join(output_dir, "vitals.csv")
    # targets_path = os.path.join(output_dir, "targets.csv")

    # results['cohort'].to_csv(cohort_path, index=False)
    # results['labs'].to_csv(labs_path, index=False)
    # results['vitals'].to_csv(vitals_path, index=False)
    # results['targets'].to_csv(targets_path, index=False)

    # print(f"\nResults saved to CSV files:")
    # print(f"  - Cohort data: {cohort_path} ({len(results['cohort'])} rows)")
    # print(f"  - Labs data: {labs_path} ({len(results['labs'])} rows)")
    # print(f"  - Vitals data: {vitals_path} ({len(results['vitals'])} rows)")
    # print(f"  - Targets data: {targets_path} ({len(results['targets'])} rows)")

    # print("\n--- Cohort Sample ---")
    # print(results['cohort'].head())

    # print("\n--- Labs Sample ---")
    # print(results['labs'].head())

    # print("\n--- Vitals Sample ---")
    # print(results['vitals'].head())

    # print("\n--- Targets Sample ---")
    # print(results['targets'].head())

    return results


# if __name__ == "__main__":
#     main()