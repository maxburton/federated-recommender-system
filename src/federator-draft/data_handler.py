import numpy as np
import random as rand
import logging
import copy
from exceptions import MissingArgumentException, InvalidShapeException


class DataHandler:
    log = logging.getLogger(__name__)

    dataset = []

    def __init__(self, filename=None, dtype=float, cols=1, ds=None):
        if filename:
            self.dataset = np.fromfile(filename, sep=" ", dtype=dtype)
            dataset_items = len(self.dataset)
            if dataset_items % cols != 0 or cols < 1:
                self.log.error("Array cannot be reshaped to %d columns! Array items = %d" % (cols, dataset_items))
                raise InvalidShapeException
            self.dataset = self.dataset.reshape(dataset_items//cols, cols)
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
        return ds[ds[:, col].argsort()]  # argsort returns an array of indices which define a new, sorted order for ds

    def extract_whole_entries(self, n, upper=None, lower=1, col=1, ds=None, delete=True):
        if ds is None:
            ds = self.dataset
        if upper is None:
            upper = n
        movie_ids = rand.sample(range(lower, upper+1), n)
        extracted = ds[np.isin(ds[:, col], movie_ids)]  # returns a boolean array to index ds
        if delete:
            # negated boolean array "deletes" extracted rows by returning a new array without those rows
            self.dataset = ds[np.invert(np.isin(ds[:, col], movie_ids))]
        return extracted

    def split_dataset_randomly(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        ds = copy.deepcopy(ds)
        return self.split_dataset(num_of_partitions, ds=np.random.shuffle(ds))

    def split_dataset_intermittently(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        weaved_ds = ds[::num_of_partitions]
        for i in range(1, num_of_partitions):
            weaved_ds = np.vstack((weaved_ds, ds[i::num_of_partitions]))
        return self.split_dataset(num_of_partitions, ds=weaved_ds)

    def split_dataset(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        return np.array_split(ds, num_of_partitions)
