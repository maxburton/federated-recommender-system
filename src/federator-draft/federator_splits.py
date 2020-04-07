from sklearn.metrics import dcg_score

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits
from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
from data_handler import DataHandler
import numpy as np
import helpers

import multiprocessing


class FederatorSplits:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, data_path=None, labels_ds=None, norm_func=None):
        self.user_id = user_id
        self.data_path = data_path
        self.labels_ds = labels_ds

        # Get golden list for splits method
        self.golden_lfm_splits, self.golden_svd_splits = GoldenList().generate_lists(self.user_id, data_path=self.data_path,
                                                                       labels_ds=labels_ds, num_of_recs=-1,
                                                                       norm_func=norm_func)

        # Normalise golden list scores
        self.golden_lfm_splits[:, 2] = helpers.scale_scores(self.golden_lfm_splits[:, 2]).flatten()
        self.golden_svd_splits[:, 2] = helpers.scale_scores(self.golden_svd_splits[:, 2]).flatten()

        self.best_dcg_score = np.inf

    def run_on_splits(self, norm_func=None):
        split_scores_lfm = IndividualSplits().run_on_splits_lfm(self.user_id, self.golden_lfm_splits, norm_func=norm_func)
        split_scores_svd = IndividualSplits().run_on_splits_svd(self.user_id, self.golden_svd_splits, norm_func=norm_func)


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
    fed = FederatorSplits(user_id, data_path=dh.get_dataset(), labels_ds="/datasets/ml-latest-small/movies.csv",
                          norm_func=norm_func)
    fed.run_on_splits(norm_func=norm_func)
