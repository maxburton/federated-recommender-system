from definitions import ROOT_DIR
import logging.config
import numpy as np
import pandas as pd
import helpers
from data_handler import DataHandler
from golden_list import GoldenList
from knn_user import KNNUser
from lightfm_alg import LightFMAlg

"""
Runs each alg on individual splits, then calculates their score against their respective golden list
"""


class IndividualSplits:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, n_subsets=5, movie_id_col=1, data_path=None):
        if data_path is None:
            ds_base_path = "/datasets/ml-latest-small"
            ds_path = ROOT_DIR + ds_base_path + "/ratings.csv"
        else:
            ds_path = ROOT_DIR + data_path

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        self.split_dataset = data.split_dataset_intermittently(n_subsets)

    # TODO: XCreate the splitsX, Xrun each alg on the splitsX, Xand for each split calculate score against golden listX
    def run_on_splits_knn(self, user_id, golden, num_of_recs=20):
        scores = []
        for i in range(len(self.split_dataset)):
            knnu = KNNUser(user_id, ds_ratings=helpers.convert_np_to_pandas(pd, self.split_dataset[i]),
                           p_thresh=0, u_thresh=0)
            split_recs = knnu.make_recommendation(num_of_recs=num_of_recs)
            r_values = helpers.get_relevant_values(split_recs, golden)
            k = 10  # (NDCG@k)
            scores.append(helpers.ndcg_at_k(r_values, k))
        return scores

    def run_on_splits_lfm(self, user_id, golden, num_of_recs=20):
        scores = []
        for i in range(len(self.split_dataset)):
            alg_warp = LightFMAlg("warp", ds=self.split_dataset[i])  # warp or bpr
            split_recs = alg_warp.generate_rec(alg_warp.model, user_id-1, num_rec=num_of_recs)  # lfm is zero indexed
            r_values = helpers.get_relevant_values(split_recs, golden)
            k = 10  # (NDCG@k)
            scores.append(helpers.ndcg_at_k(r_values, k))
        return scores


if __name__ == '__main__':
    user_id = 1
    golden_knn, golden_lfm = GoldenList().generate_lists(user_id, num_of_recs=100)
    #knn_scores = IndividualSplits().run_on_splits_knn(user_id, golden_knn)
    lfm_scores = IndividualSplits().run_on_splits_lfm(user_id, golden_lfm)
    #print("KNN: %s" % str(knn_scores))
    print("LFM: %s" % str(lfm_scores))