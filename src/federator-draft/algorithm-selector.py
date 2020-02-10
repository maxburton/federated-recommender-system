from data_handler import DataHandler
import logging.config
from definitions import ROOT_DIR
import numpy as np


class AlgorithmSelector:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    DS_PATH = ROOT_DIR + "/datasets/ml-100k/u.data"
    MOVIE_ID_COL = 1

    f_train = []  # federator train and test sets
    f_test = []
    algorithm_datasets = []

    def __init__(self):
        data = DataHandler(self.DS_PATH, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(self.MOVIE_ID_COL))
        self.f_train, self.f_test = data.extract_whole_entries_train_and_test(16, 4, col=self.MOVIE_ID_COL)

        num_of_alg_subsets = 5
        algorithm_unsplit_dataset = data.split_dataset_intermittently(num_of_alg_subsets)
        for i in range(num_of_alg_subsets):
            alg_subset = data.split_dataset_by_ratio([0.8, 0.2], ds=algorithm_unsplit_dataset[i])
            self.log.info("Algorithm %d dataset shapes: train: %s, test: %s" % (i, alg_subset[0].shape,
                                                                                alg_subset[1].shape))
            self.algorithm_datasets.append(alg_subset)


if __name__ == '__main__':
    AlgorithmSelector()
