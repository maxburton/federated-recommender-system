import numpy as np
import logging
from exceptions import MissingArgumentException
from definitions import ROOT_DIR


class DataHandler:
    log = logging.getLogger(__name__)

    dataset = []

    def __init__(self, filename=None, ds=None):
        if filename:
            self.dataset = np.fromfile(ROOT_DIR + filename, sep=" ")
        elif ds is not None:
            self.dataset = ds
        else:
            self.log.error("No filename or dataset specified!")
            raise MissingArgumentException

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def sort_dataset_by_col(self, col, ds=None):
        if ds is None:
            ds = self.dataset
        return ds[ds[:, col].argsort()]

    def extract_whole_entries(self, n, upper, lower=1, col=2, ds=None):
        if ds is None:
            ds = self.dataset
        movie_ids = np.random.randint(lower, upper+1, n)
        return ds[np.isin(ds[:, col], movie_ids)]

    def split_movielens_intermittently(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        return np.array_split(ds, num_of_partitions).T
