from definitions import ROOT_DIR
import logging.config
import numpy as np
import pandas as pd
import helpers
from data_handler import DataHandler
from golden_list import GoldenList
from knn_user import KNNUser
from lightfm_alg import LightFMAlg
from surprise_svd import SurpriseSVD
from sklearn.metrics import ndcg_score

"""
Runs each alg on individual splits, then calculates their score against their respective golden list
"""


class IndividualSplits:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, n_subsets=5, movie_id_col=1, user_id=0, data_path=None, labels_ds=None, splitting_method="even"):
        self.labels_ds = labels_ds
        self.user_id = user_id

        if data_path is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
        else:
            ds_path = ROOT_DIR + data_path

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))

        if splitting_method == "random":
            self.split_dataset, _ = data.split_dataset_randomly_ratio(n_subsets)
        elif isinstance(splitting_method, list):
            self.split_dataset = data.split_dataset_ratio_random_sort(splitting_method)
        else:
            self.split_dataset = data.split_dataset_intermittently(n_subsets)

        self.ratios = self.get_ratios()
        self.densities = self.get_densities()
        self.user_activity = self.get_user_activity()
        self.num_items_per_split = self.get_num_items_per_split()
        self.total_ratings = data.get_dataset().shape[0]

    def get_ratios(self):
        split_lens = []
        for split in self.split_dataset:
            split_lens.append(split.shape[0])

        # divide by total ratings to give a scaling factor between 0 and 1
        split_lens = np.array(split_lens)
        return split_lens / np.sum(split_lens)

    def get_densities(self):
        split_dens = []
        for split in self.split_dataset:
            # Get the number of unique users and movies (e.g. dimensions of user-item matrix)
            users = np.unique(split[:, 0])
            movies = np.unique(split[:, 1])
            unique = users.shape[0] + movies.shape[0]
            split_dens.append(unique)

        # divide by total unique items to give a density factor between 0 and 1
        split_dens = np.array(split_dens)
        split_dens = split_dens / np.sum(split_dens)
        return split_dens / self.ratios

    def get_user_activity(self):
        split_activity = []
        for split in self.split_dataset:
            users_ratings = split[split[:, 0] == self.user_id]
            split_activity.append(users_ratings.shape[0])

        # divide by total ratings to give a scaling factor between 0 and 1
        split_activity = np.array(split_activity)
        return split_activity / np.sum(split_activity)

    def get_num_items_per_split(self):
        split_count = []
        for split in self.split_dataset:
            split_count.append(split.shape[0])
        return split_count

    def run_on_splits_lfm(self, user_id, num_of_recs=20, alg="warp", norm_func=None):
        split_recs = []
        for i in range(len(self.split_dataset)):
            alg_warp = LightFMAlg(alg, ds=self.split_dataset[i], labels_ds=self.labels_ds, normalisation=norm_func)
            split_recs.append(alg_warp.generate_rec(alg_warp.model, user_id, num_rec=num_of_recs))
        return np.array(split_recs)

    def run_on_splits_svd(self, user_id, num_of_recs=20, norm_func=None):
        split_recs = []
        for i in range(len(self.split_dataset)):
            svd = SurpriseSVD(ds=self.split_dataset[i], normalisation=norm_func, save=False, load=False)
            split_recs.append(svd.get_top_n(user_id, n=num_of_recs))
        return np.array(split_recs)


if __name__ == '__main__':
    user_id = 4
    golden_lfm, golden_svd = GoldenList().generate_lists(user_id, num_of_recs=-1)
    k = 20

    lfm_scores = IndividualSplits().run_on_splits_lfm(user_id, num_of_recs=k)
    svd_scores = IndividualSplits().run_on_splits_svd(user_id, num_of_recs=k)

    print("LFM: %s" % str(lfm_scores))
    print("SVD: %s" % str(svd_scores))
