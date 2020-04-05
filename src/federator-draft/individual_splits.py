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

    def __init__(self, n_subsets=5, movie_id_col=1, data_path=None, labels_ds=None):
        self.labels_ds = labels_ds
        if data_path is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
        else:
            ds_path = ROOT_DIR + data_path

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        self.split_dataset = data.split_dataset_intermittently(n_subsets)

    #TODO: Change this to new ndcg score
    def get_ndcg_score(self, golden, split_recs, k=10):
        golden_r, predicted_r = helpers.get_relevant_values_2(split_recs, golden, k=k)
        return ndcg_score(golden_r, predicted_r, k)

    # TODO: XCreate the splitsX, Xrun each alg on the splitsX, Xand for each split calculate score against golden listX
    def run_on_splits_knn(self, user_id, golden, num_of_recs=20, k=10):
        scores = []
        for i in range(len(self.split_dataset)):
            knnu = KNNUser(user_id, ds_ratings=helpers.convert_np_to_pandas(pd, self.split_dataset[i]),
                           p_thresh=0, u_thresh=0)
            split_recs = knnu.make_recommendation(num_of_recs=num_of_recs)
            scores.append(self.get_ndcg_score(split_recs, golden, k=k))  # (NDCG@k)
        return scores

    def run_on_splits_lfm(self, user_id, golden, num_of_recs=20, k=10, norm_func=None):
        scores = []
        for i in range(len(self.split_dataset)):
            alg_warp = LightFMAlg("warp", ds=self.split_dataset[i], labels_ds=self.labels_ds, normalisation=norm_func)
            split_recs = alg_warp.generate_rec(alg_warp.model, user_id, num_rec=num_of_recs)
            scores.append(self.get_ndcg_score(split_recs, golden, k=k))  # (NDCG@k)
        return scores

    def run_on_splits_svd(self, user_id, golden, num_of_recs=20, k=10, norm_func=None):
        scores = []
        for i in range(len(self.split_dataset)):
            svd = SurpriseSVD(ds=self.split_dataset[i], normalisation=norm_func, save=False, load=False)
            split_recs = svd.get_top_n(user_id, n=num_of_recs)
            scores.append(self.get_ndcg_score(split_recs, golden, k=k))  # (NDCG@k)
        return scores


if __name__ == '__main__':
    user_id = 4
    golden_knn, golden_lfm, golden_svd = GoldenList().generate_lists(user_id, num_of_recs=-1)
    k=20
    #knn_scores = IndividualSplits().run_on_splits_knn(user_id, golden_knn, k=k)
    lfm_scores = IndividualSplits().run_on_splits_lfm(user_id, golden_lfm, k=k)
    svd_scores = IndividualSplits().run_on_splits_svd(user_id, golden_svd, k=k)
    #print("KNN: %s" % str(knn_scores))
    print("LFM: %s" % str(lfm_scores))
    print("SVD: %s" % str(svd_scores))
