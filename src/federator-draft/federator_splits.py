from sklearn.metrics import dcg_score

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits
import math
from data_handler import DataHandler
import numpy as np
import helpers

import multiprocessing


class FederatorSplits:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, alg="lfm", data_path=None, labels_ds=None, norm_func=None):
        self.user_id = user_id
        self.data_path = data_path
        self.labels_ds = labels_ds
        self.alg = alg

        self.size_ratios = None
        self.density_ratios = None
        self.user_activity_ratios = None
        self.total_ratings = None
        self.num_items_per_split = None

        # Get golden list for splits method
        self.golden_lfm_splits, self.golden_svd_splits = GoldenList().generate_lists(self.user_id, data_path=self.data_path,
                                                                       labels_ds=labels_ds, num_of_recs=-1,
                                                                       norm_func=norm_func)

        # Normalise golden list scores
        #self.golden_lfm_splits[:, 2] = helpers.scale_scores(self.golden_lfm_splits[:, 2]).flatten()
        #self.golden_svd_splits[:, 2] = helpers.scale_scores(self.golden_svd_splits[:, 2]).flatten()

        if self.alg == "lfm":
            self.golden = self.golden_lfm_splits
        else:
            self.golden = self.golden_svd_splits

        self.best_dcg_score = np.inf

    def calculate_metrics(self, federated_recs, n=10, title=""):
        federated_recs = helpers.remove_duplicate_reps(federated_recs)

        # Federated ndcg score
        golden_r, predicted_r = helpers.order_top_k_items(self.golden, federated_recs, self.log, k=n)

        # We divide by the best possible dcg score to calculate the normalised dcg score
        ndcg_score = dcg_score(golden_r, predicted_r, n) / self.best_dcg_score
        print("%s NDCG@%d Score: %.3f" % (title, n, ndcg_score))

        # Precision recall score
        pr = self.calculate_precision_recall(federated_recs, k=n)
        print("%s precision/recall@%d Score: %.3f" % (title, n, pr))

        # Precision recall scores up to k
        precisions = self.precision_up_to_k(federated_recs, k=n)

        return [title, [ndcg_score, pr], [precisions]]

    def calculate_metrics_on_splits(self, splits, n=10):
        for i in range(len(splits)):
            golden_r, predicted_r = helpers.order_top_k_items(self.golden, splits[i], self.log, k=n)
            ndcg_score = dcg_score(golden_r, predicted_r, n) / self.best_dcg_score
            print("Split %d NDCG@%d Score: %.3f" % (i, n, ndcg_score))

    """
    Note: Precision and recall are the same for our case, as the number of relevant documents is n and the
    number of retrieved documents is n
    """

    def calculate_precision_recall(self, recs, k=10):
        top_golden_titles = self.golden[:, 1][:k]
        occs = 0

        for rec in recs[:k]:
            title = rec[1]
            if np.isin(title, top_golden_titles):
                occs += 1
        return float(occs) / k

    """
    Get all precision@k scores from 1 to k
    """

    def precision_up_to_k(self, recs, k=10):
        precision_scores = []

        for i in range(k):
            pr = self.calculate_precision_recall(recs, k=i + 1)
            precision_scores.append(pr)

        return np.array(precision_scores)

    def run_on_splits(self, splitting_method=None, n_subsets=5, norm_func=None):
        isplits = IndividualSplits(n_subsets=n_subsets, user_id=self.user_id, splitting_method=splitting_method)
        self.size_ratios = isplits.ratios
        self.density_ratios = isplits.densities
        self.user_activity_ratios = isplits.user_activity
        self.total_ratings = isplits.total_ratings
        self.num_items_per_split = isplits.num_items_per_split

        if self.alg == "lfm":
            splits = isplits.run_on_splits_lfm(self.user_id, norm_func=norm_func)
        else:
            splits = isplits.run_on_splits_svd(self.user_id, norm_func=norm_func)

        return splits

    def raw_merge(self, splits, n=10):
        raw_merge = np.vstack(tuple(splits))
        raw_merge = helpers.sort_and_reset_rankings(raw_merge)
        metrics = self.calculate_metrics(raw_merge, title="Raw Merge", n=n)

        return metrics

    def db_size(self, splits, n=10):
        for i in range(splits.shape[0]):
            splits[i][:, 2] = splits[i][:, 2].astype(float) * self.size_ratios[i]
        db_size = np.vstack(tuple(splits))
        db_size = helpers.sort_and_reset_rankings(db_size)
        metrics = self.calculate_metrics(db_size, title="DB Size", n=n)

        return metrics

    def db_density(self, splits, n=10):
        for i in range(splits.shape[0]):
            splits[i][:, 2] = splits[i][:, 2].astype(float) * self.density_ratios[i]
        db_density = np.vstack(tuple(splits))
        db_density = helpers.sort_and_reset_rankings(db_density)
        metrics = self.calculate_metrics(db_density, title="DB Density", n=n)

        return metrics

    def user_activity(self, splits, n=10):
        for i in range(splits.shape[0]):
            splits[i][:, 2] = splits[i][:, 2].astype(float) * self.user_activity_ratios[i]
        user_activity = np.vstack(tuple(splits))
        user_activity = helpers.sort_and_reset_rankings(user_activity)
        metrics = self.calculate_metrics(user_activity, title="User Activity", n=n)

        return metrics

    def ecors(self, splits, n=10, threshold_mp=0.7):
        len_splits = splits.shape[0]
        b = 0.4
        t = self.num_items_per_split
        avg_t = self.total_ratings / len_splits

        all_recs = np.vstack(tuple(splits))
        all_scores = all_recs[:, 2].astype(float)
        threshold = threshold_mp * np.max(all_scores)

        dsat = 0
        at = []
        for i in range(len_splits):
            # num of items above threshold
            at_i = splits[i][splits[i][:, 2].astype(float) > threshold].shape[0]
            at.append(at_i)

            # if a split contains at least one item above threshold
            if at_i > 0:
                dsat += 1

        I = math.log((len_splits + 0.5)/dsat)/math.log(len_splits + 1.0)
        for i in range(len_splits):
            # Normalising function
            NF = 1 / t[i]

            # Calculate term frequency, T
            T = at[i]*NF / (at[i]*NF + 50*NF + (150*NF * t[i]*NF)/avg_t*NF)

            ecors = b + (1-b) * T * I
            splits[i][:, 2] = splits[i][:, 2].astype(float) * ecors

        ecors_weighted = np.vstack(tuple(splits))
        ecors_weighted = helpers.sort_and_reset_rankings(ecors_weighted)

        metrics = self.calculate_metrics(ecors_weighted, title="ECoRS", n=n)

        return metrics

    def federate(self, splitting_method=None, n_subsets=5, n=10):
        self.best_dcg_score = helpers.best_dcg_score(k=n)

        splits = self.run_on_splits(n_subsets=n_subsets, splitting_method=splitting_method, norm_func=norm_func)
        self.calculate_metrics_on_splits(splits, n=n)

        metrics = [self.raw_merge(splits, n=n),
                   self.db_size(splits, n=n),
                   self.db_density(splits, n=n),
                   self.user_activity(splits, n=n),
                   self.ecors(splits, n=n),
                   ]

        print("done")


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    norm_func = None
    ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
    dh = DataHandler(filename=ds_path)

    # Filter out users and items below threshold
    dh, surviving_users = helpers.remove_below_threshold_user_and_items(dh, u_thresh=0, i_thresh=0)

    # Get users who have at least rated at least min_ratings movies
    min_ratings_users = helpers.get_users_with_min_ratings(surviving_users, min_ratings=10)
    user_id = np.min(min_ratings_users.index.astype(int))
    user_id = 5
    fed = FederatorSplits(user_id, alg="lfm", data_path=dh.get_dataset(),
                          labels_ds="/datasets/ml-latest-small/movies.csv", norm_func=norm_func)
    splitting_method = "random"
    # splitting_method = [0.8, 0.05, 0.05, 0.05, 0.05]
    # splitting_method = [0.5, 0.3, 0.1, 0.05, 0.05]
    fed.federate(splitting_method=splitting_method, n_subsets=20, n=50)
