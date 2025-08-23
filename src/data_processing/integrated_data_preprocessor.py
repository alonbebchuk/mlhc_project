import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from .static_data_preprocessor import StaticDataPreprocessor
from .timeseries_data_preprocessor import TimeSeriesDataPreprocessor
from .data_extraction import extract_data
from .logging_utils import logger

VALIDATION_SIZE = 0.1
RANDOM_SEED = 42


class IntegratedICUPreprocessor:
    def __init__(self):
        logger.log_start("IntegratedICUPreprocessor.__init__")
        self.static_preprocessor = StaticDataPreprocessor()
        self.timeseries_preprocessor = TimeSeriesDataPreprocessor()
        logger.log_end("IntegratedICUPreprocessor.__init__")

    def _create_stratification_labels(self, targets: np.ndarray) -> np.ndarray:
        logger.log_start("IntegratedICUPreprocessor._create_stratification_labels")
        stratification_labels = []
        for row in targets:
            label = (str(int(row[0])) + str(int(row[1])) + str(int(row[2])))
            stratification_labels.append(label)
        logger.log_end("IntegratedICUPreprocessor._create_stratification_labels")
        return np.array(stratification_labels)

    def transform(self, subject_ids: List[int]) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.log_start("IntegratedICUPreprocessor.transform")
        hadm_ids, static_data, timeseries_data, timeseries_missingness, targets = extract_data(subject_ids)
        static_data, static_missingness = self.static_preprocessor.transform(static_data)
        timeseries_data = self.timeseries_preprocessor.transform(timeseries_data)
        logger.log_end("IntegratedICUPreprocessor.transform")
        return hadm_ids, static_data, static_missingness, timeseries_data, timeseries_missingness, targets

    def create_train_val_test_splits(self, initial_cohort_subject_ids: List[int], test_example_subject_ids: List[int]) -> Tuple[
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        logger.log_start("IntegratedICUPreprocessor.create_train_val_test_splits")
        cohort_hadm_ids, cohort_static_data, cohort_timeseries_data, cohort_timeseries_missingness, cohort_targets = extract_data(initial_cohort_subject_ids)
        stratification_labels = self._create_stratification_labels(cohort_targets)
        train_indices, val_indices = train_test_split(
            range(len(cohort_hadm_ids)),
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=stratification_labels
        )

        train_hadm_ids = [cohort_hadm_ids[i] for i in train_indices]
        train_static_data = cohort_static_data[train_indices]
        train_timeseries_data = cohort_timeseries_data[train_indices]
        train_timeseries_missingness = cohort_timeseries_missingness[train_indices]
        train_targets = cohort_targets[train_indices]

        val_hadm_ids = [cohort_hadm_ids[i] for i in val_indices]
        val_static_data = cohort_static_data[val_indices]
        val_timeseries_data = cohort_timeseries_data[val_indices]
        val_timeseries_missingness = cohort_timeseries_missingness[val_indices]
        val_targets = cohort_targets[val_indices]

        self.static_preprocessor, train_static_data, train_static_missingness = self.static_preprocessor.fit_transform(train_static_data)
        self.timeseries_preprocessor, train_timeseries_data = self.timeseries_preprocessor.fit_transform(train_timeseries_data)

        train_data = (train_hadm_ids, train_static_data, train_static_missingness, train_timeseries_data, train_timeseries_missingness, train_targets)

        val_static_data, val_static_missingness = self.static_preprocessor.transform(val_static_data)
        val_timeseries_data = self.timeseries_preprocessor.transform(val_timeseries_data)

        val_data = (val_hadm_ids, val_static_data, val_static_missingness, val_timeseries_data, val_timeseries_missingness, val_targets)

        test_data = self.transform(test_example_subject_ids)

        logger.log_end("IntegratedICUPreprocessor.create_train_val_test_splits")
        return train_data, val_data, test_data

    def save(self, filepath: str) -> None:
        logger.log_start("IntegratedICUPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("IntegratedICUPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'IntegratedICUPreprocessor':
        logger.log_start("IntegratedICUPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("IntegratedICUPreprocessor.load")
        return preprocessor
