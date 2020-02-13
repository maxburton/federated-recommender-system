import logging.config

import pandas as pd
from sklearn import preprocessing

import helpers
from data_handler import DataHandler
from definitions import ROOT_DIR
import numpy as np

from knn_user import KNNUser
from lightfm_alg import LightFMAlg


class AlgMapper:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, n_subsets=5, movie_id_col=1, data_path=None, split_to_train=0):
        if data_path is None:
            ds_base_path = "/datasets/ml-latest-small"
            ds_path = ROOT_DIR + ds_base_path + "/ratings.csv"
        else:
            ds_path = ROOT_DIR + data_path

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        split_dataset = data.split_dataset_intermittently(n_subsets)
        split_data = split_dataset[split_to_train]

        knnu = KNNUser(user_id, ds_ratings=helpers.convert_np_to_pandas(pd, split_data),
                       p_thresh=0, u_thresh=0)
        self.knn_recs = knnu.make_recommendation(num_of_recs=-1)  # -1 means get all available

        alg_warp = LightFMAlg("warp", ds=split_data)  # warp or bpr
        self.lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id - 1, num_rec=-1)  # lfm is zero indexed

    # normalise the data to range 0-1, after removing unique (to one algorithm) entries and sorting by movie title
    def normalise_and_trim(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        knn_sorted = self.knn_recs[self.knn_recs[:, 1].argsort()]
        lfm_sorted = self.lfm_recs[self.lfm_recs[:, 1].argsort()]

        # Remove all entries that don't exist in both lists
        knn_mask = np.in1d(knn_sorted[:, 1], lfm_sorted[:, 1])
        lfm_mask = np.in1d(lfm_sorted[:, 1], knn_sorted[:, 1])
        knn_sorted = knn_sorted[knn_mask]
        lfm_sorted = lfm_sorted[lfm_mask]
        knn_normalised_scores = 1 - min_max_scaler.fit_transform(knn_sorted[:, 2].reshape(-1, 1))  # reshape is required to fit a 1d array
        lfm_normalised_scores = min_max_scaler.fit_transform(lfm_sorted[:, 2].reshape(-1, 1))
        return knn_normalised_scores, lfm_normalised_scores

    # TODO: create a score mapping
    # TODO: perhaps invert KNN scores since a lower score is better, which is opposite to how lfm works
    # TODO: Compare to a random model to see if my model outperforms it


if __name__ == '__main__':
    user_id = 1
    mapper = AlgMapper(user_id, split_to_train=0)
    knn, lfm = mapper.normalise_and_trim()
    helpers.create_scatter_graph(["KNN", "LFM"], ["red", "blue"], knn, lfm)
