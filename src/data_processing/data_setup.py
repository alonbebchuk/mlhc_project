import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .integrated_data_preprocessor import IntegratedICUPreprocessor
from .logging_utils import logger

INITIAL_COHORT_CSV = "csvs/initial_cohort.csv"
TEST_EXAMPLE_CSV = "csvs/test_example.csv"
DATA_DIR = "data"

PREPROCESSOR_FILE = "integrated_preprocessor.pkl"
TRAIN_DATA_FILE = "train_data.pkl"
VAL_DATA_FILE = "val_data.pkl"
TEST_DATA_FILE = "test_data.pkl"


def create_data_directory():
    logger.log_start("create_data_directory")
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    logger.log_end("create_data_directory")


def load_subject_ids(csv_path: str) -> List[int]:
    logger.log_start("load_subject_ids")
    df = pd.read_csv(csv_path)
    subject_ids = df['subject_id'].tolist()
    logger.log_end("load_subject_ids")
    return subject_ids


def save_dataset(data_tuple: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], filepath: str):
    logger.log_start("save_dataset")
    hadm_ids, static_data, static_missingness, timeseries_data, timeseries_missingness, targets = data_tuple
    dataset_dict = {
        'hadm_ids': hadm_ids,
        'static_data': static_data,
        'static_missingness': static_missingness,
        'timeseries_data': timeseries_data,
        'timeseries_missingness': timeseries_missingness,
        'targets': targets
    }
    with open(filepath, 'wb') as f:
        pickle.dump(dataset_dict, f)
    logger.log_end("save_dataset")


def main():
    logger.log_start("main")
    create_data_directory()
    initial_cohort_subject_ids = load_subject_ids(INITIAL_COHORT_CSV)
    test_example_subject_ids = load_subject_ids(TEST_EXAMPLE_CSV)
    preprocessor = IntegratedICUPreprocessor()
    train_data, val_data, test_data = preprocessor.create_train_val_test_splits(
        initial_cohort_subject_ids,
        test_example_subject_ids
    )
    preprocessor_path = Path(DATA_DIR) / PREPROCESSOR_FILE
    preprocessor.save(preprocessor_path)
    train_path = Path(DATA_DIR) / TRAIN_DATA_FILE
    val_path = Path(DATA_DIR) / VAL_DATA_FILE
    test_path = Path(DATA_DIR) / TEST_DATA_FILE
    save_dataset(train_data, train_path)
    save_dataset(val_data, val_path)
    save_dataset(test_data, test_path)
    logger.log_end("main")


if __name__ == "__main__":
    main()
