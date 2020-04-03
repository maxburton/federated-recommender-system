from sklearn.metrics import dcg_score

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits
from alg_mapper import AlgMapper
from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
import numpy as np
import copy
import helpers

import multiprocessing


class Federator:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, norm_func=None):
        self.user_id = user_id
        self.golden_lfm, self.golden_svd = GoldenList().generate_lists(self.user_id, num_of_recs=-1, norm_func=norm_func)

        # Normalise golden list scores
        #self.golden_knn[:, 2] = helpers.scale_scores(self.golden_knn[:, 2]).flatten()
        self.golden_lfm[:, 2] = helpers.scale_scores(self.golden_lfm[:, 2]).flatten()
        self.golden_svd[:, 2] = helpers.scale_scores(self.golden_svd[:, 2]).flatten()

        self.best_dcg_score = np.inf

    # TODO: Should probably remove or move this to a new class
    def run_on_splits(self, norm_func=None):
        #split_scores_knn = IndividualSplits().run_on_splits_knn(self.user_id, self.golden_knn)
        split_scores_lfm = IndividualSplits().run_on_splits_lfm(self.user_id, self.golden_lfm, norm_func=norm_func)
        split_scores_svd = IndividualSplits().run_on_splits_svd(self.user_id, self.golden_svd, norm_func=norm_func)

    """
    Merges scores together based on their scores after mapping from one algorithm to the other
    """
    def merge_scores_by_mapping(self, lfm_recs, svd_recs, n=10):
        # Merge and sort
        federated_recs = np.concatenate((lfm_recs, svd_recs), axis=0)
        federated_recs = federated_recs[np.argsort(federated_recs[:, 2])][::-1]  # sort in descending order of score
        federated_recs[:, 0] = np.arange(1, federated_recs.shape[0]+1)  # Reset rankings
        federated_recs_truncated = federated_recs[:n]

        # Print the top n results
        helpers.pretty_print_results(self.log, federated_recs_truncated, self.user_id)

        # Separate algorithm scores for graph mapping
        federated_lfm_recs_truncated = federated_recs_truncated[federated_recs_truncated[:, 3] == "lfm"]
        federated_svd_recs_truncated = federated_recs_truncated[federated_recs_truncated[:, 3] == "svd"]
        helpers.create_scatter_graph("Federated Results", "Ranking", "Normalised Score",
                                     ["LFM", "SVD"], ["red", "blue"],
                                     federated_lfm_recs_truncated[:, 2].astype(float),
                                     federated_svd_recs_truncated[:, 2].astype(float),
                                     x=[federated_lfm_recs_truncated[:, 0].astype(int),
                                        federated_svd_recs_truncated[:, 0].astype(int)])

        # Federated ndcg score
        golden_r_lfm, predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm, federated_recs, self.log, k=n)
        golden_r_svd, predicted_r_svd = helpers.order_top_k_items(self.golden_svd, federated_recs, self.log, k=n)

        # We divide by the best possible dcg score to calculate the normalised dcg score
        ndcg_score_lfm = dcg_score(golden_r_lfm, predicted_r_lfm, n) / self.best_dcg_score
        ndcg_score_svd = dcg_score(golden_r_svd, predicted_r_svd, n) / self.best_dcg_score

        print("LFM NDCG@%d Score: %.5f" % (n, ndcg_score_lfm))
        print("SVD NDCG@%d Score: %.5f" % (n, ndcg_score_svd))

        return federated_recs

    """
    Weaves the top scoring items from each algorithm together, intermittently.
    
    e.g. If the top item is from LFM, then the next item will be the top item from SVD. #3 will be from LFM,
    #4 from SVD, and so on.
    
    The top item is always from the first algorithm (LFM).
    """
    def weave_scores_before_mapping(self, lfm_unmapped, svd_unmapped, n=10):
        weaved_recs = []
        max_i = n+1//2  # ensures this works for odd n
        for i in range(max_i):
            weaved_recs.append(lfm_unmapped[i])
            if i*2 <= n:  # For odd n, will not append this for the final loop
                weaved_recs.append(svd_unmapped[i+1])
        weave_b_golden_r_lfm, weave_b_predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm,
                                                                                  helpers.pick_random(
                                                                                      np.array(weaved_recs), n),
                                                                                  self.log, k=n, in_order=True)
        weave_b_golden_r_svd, weave_b_predicted_r_svd = helpers.order_top_k_items(self.golden_svd,
                                                                                  helpers.pick_random(
                                                                                      np.array(weaved_recs), n),
                                                                                  self.log, k=n, in_order=True)

        print("LFM weave before mapping NDCG@%d Score: %.5f" % (
            n, dcg_score(weave_b_golden_r_lfm, weave_b_predicted_r_lfm, n) / self.best_dcg_score))
        print("SVD weave before mapping NDCG@%d Score: %.5f" % (
            n, dcg_score(weave_b_golden_r_svd, weave_b_predicted_r_svd, n) / self.best_dcg_score))

    """
    Weaves the top scoring items from each algorithm together, intermittently.

    e.g. If the top item is from LFM, then the next item will be the top item from SVD. #3 will be from LFM,
    #4 from SVD, and so on.

    The top item is determined by whichever algorithm has the highest scored item after mapping.
    """
    def weave_scores_after_mapping(self, lfm, svd, n=10):
        # Get the algorithm with the highest score after mapping
        lfm_scores_gt_svd = lfm[:, 2].astype(float).max() > svd[:, 2].astype(float).max()
        top_alg = lfm if lfm_scores_gt_svd else svd
        other_alg = svd if lfm_scores_gt_svd else lfm

        weaved_recs = []
        max_i = n+1//2  # ensures this works for odd n
        for i in range(max_i):
            weaved_recs.append(top_alg[i])
            if i*2 <= n:  # For odd n, will not append this for the final loop
                weaved_recs.append(other_alg[i+1])
        weave_b_golden_r_lfm, weave_b_predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm,
                                                                                  helpers.pick_random(
                                                                                      np.array(weaved_recs), n),
                                                                                  self.log, k=n, in_order=True)
        weave_b_golden_r_svd, weave_b_predicted_r_svd = helpers.order_top_k_items(self.golden_svd,
                                                                                  helpers.pick_random(
                                                                                      np.array(weaved_recs), n),
                                                                                  self.log, k=n, in_order=True)

        print("LFM weave before mapping NDCG@%d Score: %.5f" % (
            n, dcg_score(weave_b_golden_r_lfm, weave_b_predicted_r_lfm, n) / self.best_dcg_score))
        print("SVD weave before mapping NDCG@%d Score: %.5f" % (
            n, dcg_score(weave_b_golden_r_svd, weave_b_predicted_r_svd, n) / self.best_dcg_score))

    """
    A baseline for NDCG, which picks randomly from a list of federated recommendations.
    """
    def pick_random_baseline(self, federated_recs, n=10):
        # Pick random recs from the federated recs, to show the importance of the mapper
        rand_golden_r_lfm, rand_predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm,
                                                                            helpers.pick_random(federated_recs, n),
                                                                            self.log, k=n, in_order=True)
        rand_golden_r_svd, rand_predicted_r_svd = helpers.order_top_k_items(self.golden_svd,
                                                                            helpers.pick_random(federated_recs, n),
                                                                            self.log, k=n, in_order=True)

        print("LFM random baseline NDCG@%d Score: %.5f" % (
            n, dcg_score(rand_golden_r_lfm, rand_predicted_r_lfm, n) / self.best_dcg_score))
        print("SVD random baseline NDCG@%d Score: %.5f" % (
            n, dcg_score(rand_golden_r_svd, rand_predicted_r_svd, n) / self.best_dcg_score))

    """
    A baseline for NDCG, which replaces one algorithm's contributions with random items from that algorithm's
    entire contributions.
    
    e.g. if the federated list has 30 items from the SVD algorithm, those 30 items will be replace with randomly
    ranked SVD items.
    """
    def replace_random_baseline(self, federated_recs, n=10):
        # Get all lfm and svd recs
        federated_lfm_recs = federated_recs[federated_recs[:, 3] == "lfm"]
        federated_svd_recs = federated_recs[federated_recs[:, 3] == "svd"]

        federated_lfm_recs_truncated = federated_lfm_recs[federated_lfm_recs[:, 0].astype(float) <= n]
        federated_svd_recs_truncated = federated_svd_recs[federated_svd_recs[:, 0].astype(float) <= n]

        # Replace lfm recs in federated list with random lfm recs (and vice versa for svd)
        random_lfm_recs = helpers.pick_random(federated_lfm_recs, n - federated_svd_recs_truncated.shape[0])
        svd_with_random_lfm = np.concatenate((federated_svd_recs, random_lfm_recs), axis=0)
        svd_with_random_lfm = svd_with_random_lfm[np.argsort(svd_with_random_lfm[:, 2])][::-1]  # sort

        random_svd_recs = helpers.pick_random(federated_svd_recs, n - federated_lfm_recs_truncated.shape[0])
        lfm_with_random_svd = np.concatenate((federated_lfm_recs, random_svd_recs), axis=0)
        lfm_with_random_svd = lfm_with_random_svd[np.argsort(lfm_with_random_svd[:, 2])][::-1]

        golden_r_lfm_rand_svd, predicted_r_lfm_rand_svd = helpers.order_top_k_items(self.golden_lfm,
                                                                                    lfm_with_random_svd, self.log, k=n,
                                                                                    in_order=True)
        golden_r_svd_rand_lfm, predicted_r_svd_rand_lfm = helpers.order_top_k_items(self.golden_svd,
                                                                                    svd_with_random_lfm, self.log, k=n,
                                                                                    in_order=True)

        print("LFM with random SVD baseline NDCG@%d Score: %.5f" % (n, dcg_score(golden_r_lfm_rand_svd,
                                                                                 predicted_r_lfm_rand_svd,
                                                                                 n) / self.best_dcg_score))
        print("SVD with random LFM baseline NDCG@%d Score: %.5f" % (n, dcg_score(golden_r_svd_rand_lfm,
                                                                                 predicted_r_svd_rand_lfm,
                                                                                 n) / self.best_dcg_score))

    def federate_results(self, n, norm_func=None, reverse_mapping=False):
        # TODO: check performance difference between mapping svd to lfm AND lfm to svd
        # Normalise and map scores (default mapping is svd -> lfm)
        mapper = AlgMapper(self.user_id, split_to_train=0, norm_func=norm_func)
        lfm, svd = mapper.normalise_and_trim()
        if not reverse_mapping:
            model = mapper.learn_mapping(svd, lfm)
        else:
            model = mapper.learn_mapping(lfm, svd)

        splits = mapper.untrained_data
        split_to_predict = 0
        dataset = splits[split_to_predict]

        # Get LFM's recs
        alg_warp = LightFMAlg("warp", ds=dataset, normalisation=norm_func)
        lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id, num_rec=-1)
        lfm_recs = np.c_[lfm_recs, np.full(lfm_recs.shape[0], "lfm")]  # append new column of "lfm" to recs

        # Get Surprise's SVD recs
        svd_split_filename = "/svd_split_{0}.npy".format(split_to_predict)
        svd = SurpriseSVD(ds=dataset, normalisation=norm_func, save_filename=svd_split_filename, load_filename=svd_split_filename)
        svd.print_user_favourites(self.user_id)
        svd_recs = svd.get_top_n(self.user_id, n=-1)
        svd_recs = np.c_[svd_recs, np.full(svd_recs.shape[0], "svd")]  # append new column of "svd" to recs

        # Scale scores but don't map them (reshape to convert back to 1d row array)
        lfm_recs_unmapped = copy.deepcopy(lfm_recs)
        svd_recs_unmapped = copy.deepcopy(svd_recs)
        lfm_recs_unmapped[:, 2] = helpers.scale_scores(lfm_recs[:, 2]).reshape(-1)
        svd_recs_unmapped[:, 2] = helpers.scale_scores(svd_recs[:, 2]).reshape(-1)

        # Scale scores and map one to another (reshape to convert back to 1d row array)
        if not reverse_mapping:
            lfm_recs[:, 2] = lfm_recs_unmapped[:, 2].reshape(-1)
            svd_recs[:, 2] = model.predict(helpers.scale_scores(svd_recs[:, 2])).reshape(-1)
        else:
            svd_recs[:, 2] = svd_recs_unmapped[:, 2].reshape(-1)
            lfm_recs[:, 2] = model.predict(helpers.scale_scores(lfm_recs[:, 2])).reshape(-1)

        # Get the maximum possible dcg score, if the rankings were to be perfectly ranked
        self.best_dcg_score = helpers.best_dcg_score(n)

        # Run all merging algorithms
        federated_recs = self.merge_scores_by_mapping(lfm_recs, svd_recs, n=n)
        self.weave_scores_before_mapping(lfm_recs_unmapped, svd_recs_unmapped, n=n)
        self.weave_scores_after_mapping(lfm_recs, svd_recs, n=n)
        self.pick_random_baseline(federated_recs, n=n)
        self.replace_random_baseline(federated_recs, n=n)

        print("test")
        # TODO: Implement precision and recall and perhaps accuracy scores (would this work/tell me anything)


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    user_id = 5
    norm_func = helpers.gaussian_normalisation
    fed = Federator(user_id, norm_func=norm_func)
    fed.federate_results(50, reverse_mapping=False, norm_func=norm_func)
