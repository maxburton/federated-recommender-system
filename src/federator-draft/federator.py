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

    def run_on_splits(self):
        golden_knn, golden_lfm, golden_svd = GoldenList().generate_lists(self.user_id)
        #split_scores_knn = IndividualSplits().run_on_splits_knn(self.user_id, golden_knn)
        split_scores_lfm = IndividualSplits().run_on_splits_lfm(self.user_id, golden_lfm)
        split_scores_svd = IndividualSplits().run_on_splits_svd(self.user_id, golden_svd)

    def federate_results(self, n):
        mapper = AlgMapper(self.user_id, split_to_train=0)
        svd, lfm = mapper.normalise_and_trim()
        model = mapper.learn_mapping(svd, lfm)

        splits = mapper.untrained_data
        split_to_predict = 0
        dataset = splits[split_to_predict]

        # Get LFM's recs
        alg_warp = LightFMAlg("warp", ds=dataset)
        lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id - 1, num_rec=n)  # lfm is zero indexed
        lfm_recs = np.c_[lfm_recs, np.full(lfm_recs.shape[0], "lfm")]  # append new column of "lfm" to recs

        # Get Surprise's SVD recs
        svd = SurpriseSVD()
        svd.print_user_favourites(self.user_id)
        svd_recs = svd.get_top_n(self.user_id, n=n)
        svd_recs = np.c_[svd_recs, np.full(svd_recs.shape[0], "svd")]  # append new column of "svd" to recs

        # Scale scores
        lfm_recs[:, 2] = mapper.scale_scores(lfm_recs[:, 2]).reshape(-1)  # reshape to convert back to 1d row array
        svd_recs[:, 2] = model.predict(mapper.scale_scores(svd_recs[:, 2])).reshape(-1)

        # Merge and sort
        federated_recs = np.concatenate((lfm_recs, svd_recs), axis=0)
        federated_recs = federated_recs[np.argsort(federated_recs[:, 2])][::-1]  # sort in descending order of score
        federated_recs[:, 0] = np.arange(1, n*2+1)
        federated_recs = federated_recs[:n]  # Take the top n results

        helpers.pretty_print_results(self.log, federated_recs, self.user_id)

        # Separate algorithm scores for graph mapping
        federated_svd_recs = federated_recs[federated_recs[:, 3] == "svd"]
        federated_lfm_recs = federated_recs[federated_recs[:, 3] == "lfm"]
        helpers.create_scatter_graph("Federated Results", "Ranking", "Normalised Score",
                                     ["SVD", "LFM"], ["red", "blue"],
                                     federated_svd_recs[:, 2].astype(float),
                                     federated_lfm_recs[:, 2].astype(float),
                                     x=[federated_svd_recs[:, 0].astype(int), federated_lfm_recs[:, 0].astype(int)])


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    user_id = 4
    fed = Federator(user_id)
    fed.federate_results(20)
