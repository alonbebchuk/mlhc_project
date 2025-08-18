# MLHC Final Project - MIMIC-III Data Extraction

This project provides tools for extracting and processing data from the MIMIC-III database for machine learning in healthcare applications.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mlhc-project
```

## MIMIC-III Data Access Setup

To access the MIMIC-III dataset, you need to create a shortcut in your Google Drive:

1. Navigate to your Google Drive
2. Go to `My Drive/MIMIC-III` (create this folder if it doesn't exist)
3. Visit https://drive.google.com/drive/folders/11HfvZC7kx4ha5syrMLXJHkKF_5pL1b0S
4. Click on "Organize" → "Add shortcut"
5. Select the `My Drive/MIMIC-III` folder as the destination

This will create a shortcut to the MIMIC-III dataset in your Drive, allowing the data extraction script to access the necessary files.

## Google Drive Authentication Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API
4. Create credentials (OAuth 2.0 Client IDs)
   - Application type: "Desktop application"
5. Download the credentials file and save it as `credentials.json` in the project root

## File Structure
```
Final Project/
├── credentials.json          # Your Google API credentials (you provide this)
├── token.pickle             # Auto-generated authentication token
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment file
├── data/
│   └── mimiciii.duckdb     # Downloaded MIMIC database
├── csvs/                   # Metadata CSV files
│   ├── initial_cohort.csv
│   ├── labs_metadata.csv
│   └── vital_metadata.csv
└── src/
    ├── data_extraction.py  # Main data extraction script
    ├── e2e_test_data_extraction.py
    ├── unit_test_data_extraction_comprehensive.py
    └── unseen_data_evaluation.py
```

## Usage

### Running Data Extraction
```bash
python src/data_extraction.py
```

### Running Tests
```bash
# Run comprehensive unit tests
pytest src/unit_test_data_extraction_comprehensive.py -v

# Run end-to-end tests
python src/e2e_test_data_extraction.py
```

## Authentication Flow

On first run, the script will:
1. Check for `credentials.json` in the project root
2. Open a browser window for Google authentication
3. Save authentication token to `token.pickle` for future use
4. Download the MIMIC-III database from Google Drive (if not already present)

## Notes

- The authentication files (`credentials.json`, `token.pickle`) are stored in the project root
- The MIMIC-III database is downloaded to `data/mimiciii.duckdb` and cached locally
- All Google Drive API calls use read-only permissions