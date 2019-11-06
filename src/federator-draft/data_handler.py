import numpy as np
import random as rand
import logging
import copy
import exceptions as ex


class DataHandler:
    log = logging.getLogger(__name__)

    dataset = []

    def __init__(self, filename=None, dtype=float, cols=1, ds=None):
        if filename:
            self.dataset = np.fromfile(filename, sep=" ", dtype=dtype)
            dataset_items = len(self.dataset)
            if dataset_items % cols != 0 or cols < 1:
                self.log.error("Array cannot be reshaped to %d columns! Array items = %d" % (cols, dataset_items))
                raise ex.InvalidShapeException
            self.dataset = self.dataset.reshape(dataset_items//cols, cols)
        elif ds is not None:
            self.dataset = ds
        else:
            self.log.error("No filename or dataset specified!")
            raise ex.MissingArgumentException

    def get_dataset(self):
        return self.dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def sort_dataset_by_col(self, col, ds=None):
        if ds is None:
            ds = self.dataset
        return ds[ds[:, col].argsort()]  # argsort returns an array of indices which define a new, sorted order for ds

    def sort_dataset_randomly(self, ds=None):
        if ds is None:
            ds = self.dataset
        ds = copy.deepcopy(ds)
        return np.random.shuffle(ds)

    def sort_dataset_intermittently(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        weaved_ds = ds[::num_of_partitions]
        for i in range(1, num_of_partitions):
            weaved_ds = np.vstack((weaved_ds, ds[i::num_of_partitions]))
        return weaved_ds

    def extract_whole_entries(self, n, lower=1, upper=None, col=1, ds=None, delete=True):
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
        return self.split_dataset_evenly(num_of_partitions, ds=self.sort_dataset_randomly(ds))

    def split_dataset_intermittently(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        return self.split_dataset_evenly(num_of_partitions,
                                         ds=self.sort_dataset_intermittently(num_of_partitions, ds=ds))

    def split_dataset_evenly(self, num_of_partitions, ds=None):
        if ds is None:
            ds = self.dataset
        return np.array_split(ds, num_of_partitions)

    def split_dataset_by_ratio(self, num_of_partitions, splits, ds=None):
        splits = np.array(splits)
        if ds is None:
            ds = self.dataset
        if np.sum(splits) != 1.0:
            self.log.error("Split ratios add up to %.2f but should be 1.0!" % np.sum(splits))
            raise ex.InvalidRatioSumException
        if len(splits) != num_of_partitions:
            self.log.error("Number of elements in splits (%d) must be equal to the number of partitions (%d)!" %
                           (num_of_partitions, len(splits)))
            raise ex.InvalidRatioIndicesException
        splits = np.round(splits * len(ds)).astype(int)  # multiply each split float to its respective ds index
        split_array = ds[:splits[0]]
        for i in range(1, num_of_partitions):  # split dataset according to the splits ratios defined
            current_index = np.sum(splits[:i])
            split_array = np.vstack((split_array, ds[current_index:(current_index + splits[i])]))
        return split_array
