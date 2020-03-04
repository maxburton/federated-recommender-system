from sklearn.metrics import ndcg_score

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits
from alg_mapper import AlgMapper
from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
import numpy as np
import helpers

import multiprocessing


class Federator:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id):
        self.user_id = user_id
        self.golden_knn, self.golden_lfm, self.golden_svd = GoldenList().generate_lists(self.user_id, num_of_recs=-1)

        # Normalise golden list scores
        self.golden_knn[:, 2] = helpers.scale_scores(self.golden_knn[:, 2]).flatten()
        self.golden_lfm[:, 2] = helpers.scale_scores(self.golden_lfm[:, 2]).flatten()
        self.golden_svd[:, 2] = helpers.scale_scores(self.golden_svd[:, 2]).flatten()

    # TODO: Should probably remove or move this to a new class
    def run_on_splits(self):
        golden_knn, golden_lfm, golden_svd = GoldenList().generate_lists(self.user_id, num_of_recs=-1)
        #split_scores_knn = IndividualSplits().run_on_splits_knn(self.user_id, golden_knn)
        split_scores_lfm = IndividualSplits().run_on_splits_lfm(self.user_id, golden_lfm)
        split_scores_svd = IndividualSplits().run_on_splits_svd(self.user_id, golden_svd)

    def federate_results(self, n):
        # TODO: check performance difference between mapping svd to lfm AND lfm to svd
        # Normalise and map scores
        mapper = AlgMapper(self.user_id, split_to_train=0)
        lfm, svd = mapper.normalise_and_trim()
        model = mapper.learn_mapping(lfm, svd)

        splits = mapper.untrained_data
        split_to_predict = 0
        dataset = splits[split_to_predict]

        # Get LFM's recs
        alg_warp = LightFMAlg("warp", ds=dataset)
        lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id, num_rec=n)
        lfm_recs = np.c_[lfm_recs, np.full(lfm_recs.shape[0], "lfm")]  # append new column of "lfm" to recs

        # Get Surprise's SVD recs
        svd = SurpriseSVD()
        svd.print_user_favourites(self.user_id)
        svd_recs = svd.get_top_n(self.user_id, n=n)
        svd_recs = np.c_[svd_recs, np.full(svd_recs.shape[0], "svd")]  # append new column of "svd" to recs

        # Scale scores
        lfm_recs[:, 2] = helpers.scale_scores(lfm_recs[:, 2]).reshape(-1)  # reshape to convert back to 1d row array
        svd_recs[:, 2] = model.predict(helpers.scale_scores(svd_recs[:, 2])).reshape(-1)

        # Merge and sort
        federated_recs = np.concatenate((lfm_recs, svd_recs), axis=0)
        federated_recs = federated_recs[np.argsort(federated_recs[:, 2])][::-1]  # sort in descending order of score
        federated_recs[:, 0] = np.arange(1, n*2+1)
        federated_recs = federated_recs[:n]  # Take the top n results

        helpers.pretty_print_results(self.log, federated_recs, self.user_id)

        # Separate algorithm scores for graph mapping
        federated_lfm_recs = federated_recs[federated_recs[:, 3] == "lfm"]
        federated_svd_recs = federated_recs[federated_recs[:, 3] == "svd"]
        helpers.create_scatter_graph("Federated Results", "Ranking", "Normalised Score",
                                     ["LFM", "SVD"], ["red", "blue"],
                                     federated_lfm_recs[:, 2].astype(float),
                                     federated_svd_recs[:, 2].astype(float),
                                     x=[federated_lfm_recs[:, 0].astype(int), federated_svd_recs[:, 0].astype(int)])

        golden_r_lfm, predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm, federated_recs, self.log, k=n)
        golden_r_svd, predicted_r_svd = helpers.order_top_k_items(self.golden_svd, federated_recs, self.log, k=n)

        print("LFM NDCG@%d Score: %.5f" % (n, ndcg_score(predicted_r_lfm, golden_r_lfm, n)))
        print("SVD NDCG@%d Score: %.5f" % (n, ndcg_score(predicted_r_svd, golden_r_svd, n)))

        # TODO: Implement precision and recall and perhaps accuracy scores


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    user_id = 5
    fed = Federator(user_id)
    fed.federate_results(20)
