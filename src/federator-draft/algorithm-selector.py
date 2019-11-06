from data_handler import DataHandler
import logging.config
from definitions import ROOT_DIR
import numpy as np


class AlgorithmSelector:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    ML100K_PATH = ROOT_DIR + "/datasets/ml-100k/u.data"

    federator_training_set = []
    federator_test_set = []
    algorithm_datasets = []

    def __init__(self):
        data = DataHandler(self.ML100K_PATH, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(1))
        self.federator_training_set = data.extract_whole_entries(16)
        self.federator_test_set = data.extract_whole_entries(4, lower=17, upper=20)

        num_of_alg_subsets = 5
        algorithm_unsplit_dataset = data.split_dataset_intermittently(num_of_alg_subsets)
        self.algorithm_datasets = data.split_dataset_by_ratio(2, [0.8, 0.2], ds=algorithm_unsplit_dataset[0]) # TODO: Figure out how to append arrays of different row counts
        for i in range(1, num_of_alg_subsets):
            alg_subset = data.split_dataset_by_ratio(2, [0.8, 0.2], ds=algorithm_unsplit_dataset[i])
            self.algorithm_datasets = np.append(self.algorithm_datasets, alg_subset, axis=0)
        print(self.algorithm_datasets)


if __name__ == '__main__':
    AlgorithmSelector()
