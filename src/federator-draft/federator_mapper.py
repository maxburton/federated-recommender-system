from sklearn.metrics import dcg_score

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from alg_mapper import AlgMapper
from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
from data_handler import DataHandler
import numpy as np
import copy
import helpers
import matplotlib.pyplot as plt

import multiprocessing


class FederatorMapper:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, data_path=None, labels_ds=None, norm_func=None, reverse_mapping=False):
        self.user_id = user_id
        self.data_path = data_path
        self.labels_ds = labels_ds
        self.norm_func = norm_func

        # Get mapping model (default mapping is svd -> lfm)
        mapper = AlgMapper(self.user_id, data_path=self.data_path, n_subsets=2, norm_func=self.norm_func)
        lfm_normalised, svd_normalised = mapper.normalise_and_trim()
        if not reverse_mapping:
            self.model = mapper.learn_mapping(svd_normalised, lfm_normalised)
        else:
            self.model = mapper.learn_mapping(lfm_normalised, svd_normalised)

        splits = mapper.untrained_data
        self.dataset = np.vstack(splits)

        # Golden list for mapping method
        self.golden_lfm_mapper, self.golden_svd_mapper = GoldenList().generate_lists(self.user_id,
                                                                                     data_path=self.dataset,
                                                                                     labels_ds=self.labels_ds,
                                                                                     num_of_recs=-1,
                                                                                     norm_func=self.norm_func)

        # Normalise golden list scores
        self.golden_lfm_mapper[:, 2] = helpers.scale_scores(self.golden_lfm_mapper[:, 2]).flatten()
        self.golden_svd_mapper[:, 2] = helpers.scale_scores(self.golden_svd_mapper[:, 2]).flatten()

        self.best_dcg_score = np.inf

    def remove_duplicate_reps(self, recs):
        titles = []
        unique_recs = []
        for rec in recs:
            title = rec[1]
            if title not in titles:
                unique_recs.append(rec)
            titles.append(title)
        return np.array(unique_recs)

    def calculate_metrics(self, federated_recs, n=10, title=""):
        federated_recs = self.remove_duplicate_reps(federated_recs)

        # Federated ndcg score
        golden_r_lfm, predicted_r_lfm = helpers.order_top_k_items(self.golden_lfm_mapper, federated_recs, self.log, k=n)
        golden_r_svd, predicted_r_svd = helpers.order_top_k_items(self.golden_svd_mapper, federated_recs, self.log, k=n)

        # We divide by the best possible dcg score to calculate the normalised dcg score
        ndcg_score_lfm = dcg_score(golden_r_lfm, predicted_r_lfm, n) / self.best_dcg_score
        ndcg_score_svd = dcg_score(golden_r_svd, predicted_r_svd, n) / self.best_dcg_score

        pr_lfm, pr_svd = self.calculate_precision_recall(federated_recs, n=n)

        print("LFM %s NDCG@%d Score: %.3f" % (title, n, ndcg_score_lfm))
        print("SVD %s NDCG@%d Score: %.3f" % (title, n, ndcg_score_svd))

        print("LFM %s precision/recall@%d Score: %.3f" % (title, n, pr_lfm))
        print("SVD %s precision/recall@%d Score: %.3f" % (title, n, pr_svd))

        return ndcg_score_lfm, ndcg_score_svd, pr_lfm, pr_svd

    """
    Note: Precision and recall are the same for our case, as the number of relevant documents is n and the
    number of retrieved documents is n
    """
    def calculate_precision_recall(self, recs, n=10):
        top_golden_lfm_titles = self.golden_lfm_mapper[:, 1][:n]
        top_golden_svd_titles = self.golden_svd_mapper[:, 1][:n]

        lfm_occs = 0
        svd_occs = 0

        for rec in recs:
            title = rec[1]
            if np.isin(title, top_golden_lfm_titles):
                lfm_occs += 1
            if np.isin(title, top_golden_svd_titles):
                svd_occs += 1

        return float(lfm_occs)/n, float(svd_occs)/n

    """
    Merges scores together based on their scores after mapping from one algorithm to the other
    """
    def merge_scores_by_mapping(self, lfm_recs, svd_recs, n=10):
        # Merge and sort
        federated_recs = np.concatenate((lfm_recs, svd_recs), axis=0)
        federated_recs = federated_recs[np.argsort(federated_recs[:, 2])][::-1]  # sort in descending order of score
        federated_recs = self.remove_duplicate_reps(federated_recs)  # remove the second copy of items that are recommended by both algs
        federated_recs[:, 0] = np.arange(1, federated_recs.shape[0]+1)  # Reset rankings
        federated_recs_truncated = federated_recs[:n]

        # Print the top n results
        self.log.info("Federated results:")
        helpers.pretty_print_results(self.log, federated_recs_truncated, self.user_id)

        # Separate algorithm scores for graph mapping
        federated_lfm_recs_truncated = federated_recs_truncated[federated_recs_truncated[:, 3] == "lfm"]
        federated_svd_recs_truncated = federated_recs_truncated[federated_recs_truncated[:, 3] == "svd"]
        helpers.create_scatter_graph("Federated Results", "Ranking", "Normalised Score",
                                     ["LFM", "SVD"], ["blue", "orange"],
                                     federated_lfm_recs_truncated[:, 2].astype(float),
                                     federated_svd_recs_truncated[:, 2].astype(float),
                                     x=[federated_lfm_recs_truncated[:, 0].astype(int),
                                        federated_svd_recs_truncated[:, 0].astype(int)])

        title = "Mapped"
        ndcg_lfm, ndcg_svd, pr_lfm, pr_svd = self.calculate_metrics(federated_recs, n=n, title=title)

        return federated_recs, ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

    """
    Merges scores based on their normalised scores, unmapped
    """
    def merge_by_raw_scores(self, lfm_unmapped, svd_unmapped, n=10):
        # Merge and sort
        raw_merged_recs = np.concatenate((lfm_unmapped, svd_unmapped), axis=0)
        raw_merged_recs = raw_merged_recs[np.argsort(raw_merged_recs[:, 2])][::-1]  # sort in descending order of score
        raw_merged_recs[:, 0] = np.arange(1, raw_merged_recs.shape[0]+1)  # Reset rankings

        title = "Raw Merge"
        ndcg_lfm, ndcg_svd, pr_lfm, pr_svd = self.calculate_metrics(raw_merged_recs, n=n, title=title)

        return ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

    """
    Weaves the top scoring items from each algorithm together, intermittently.
    
    e.g. If the top item is from LFM, then the next item will be the top item from SVD. #3 will be from LFM,
    #4 from SVD, and so on.
    
    The top item is always from the first algorithm (LFM).
    """
    def weave_scores_before_mapping(self, lfm_unmapped, svd_unmapped, n=10):
        weaved_recs = []
        for i in range(n*2):
            try:
                weaved_recs.append(lfm_unmapped[i])
                weaved_recs.append(svd_unmapped[i + 1])
            except IndexError:
                self.log.warning("You don't have that many recommendations! Try lowering n")

        title = "Unmapped Weave"
        ndcg_lfm, ndcg_svd, pr_lfm, pr_svd = self.calculate_metrics(weaved_recs, n=n, title=title)

        return ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

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
        for i in range(n*2):
            try:
                weaved_recs.append(top_alg[i])
                weaved_recs.append(other_alg[i + 1])
            except IndexError:
                self.log.warning("You don't have that many recommendations! Try lowering n")

        title = "Mapped Weave"
        ndcg_lfm, ndcg_svd, pr_lfm, pr_svd = self.calculate_metrics(weaved_recs, n=n, title=title)

        return ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

    """
    A baseline for NDCG, which picks randomly from a list of federated recommendations.
    """
    def pick_random_baseline(self, federated_recs, n=10):
        # Pick random recs from the federated recs, to show the importance of the mapper
        title = "Random"
        ndcg_lfm, ndcg_svd, pr_lfm, pr_svd = self.calculate_metrics(helpers.pick_random(federated_recs, n=n), n=n, title=title)

        return ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

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

        # Truncate recs so there are only n recs
        federated_lfm_recs_truncated = federated_lfm_recs[federated_lfm_recs[:, 0].astype(float) <= n]
        federated_svd_recs_truncated = federated_svd_recs[federated_svd_recs[:, 0].astype(float) <= n]

        # Replace lfm recs in federated list with random lfm recs (and vice versa for svd)
        random_lfm_recs = helpers.pick_random(federated_lfm_recs, n - federated_svd_recs_truncated.shape[0])
        svd_with_random_lfm = np.concatenate((federated_svd_recs, random_lfm_recs), axis=0)
        svd_with_random_lfm = svd_with_random_lfm[np.argsort(svd_with_random_lfm[:, 2])][::-1]  # sort

        random_svd_recs = helpers.pick_random(federated_svd_recs, n - federated_lfm_recs_truncated.shape[0])
        lfm_with_random_svd = np.concatenate((federated_lfm_recs, random_svd_recs), axis=0)
        lfm_with_random_svd = lfm_with_random_svd[np.argsort(lfm_with_random_svd[:, 2])][::-1]

        title = "One Alg Random"
        ndcg_lfm, _, pr_lfm, _ = self.calculate_metrics(helpers.pick_random(lfm_with_random_svd, n=n), n=n, title=title)
        _, ndcg_svd, _, pr_svd = self.calculate_metrics(helpers.pick_random(svd_with_random_lfm, n=n), n=n, title=title)

        return ndcg_lfm, ndcg_svd, pr_lfm, pr_svd, title

    """
    A baseline for NDCG, which compares golden list of one alg to the recs from both golden lists.
    """
    def alg_on_alg(self, recs, n=10, title="Alg on Alg"):
        rec_on_lfm, rec_on_svd, pr_lfm, pr_svd = self.calculate_metrics(recs, n=n, title=title)

        return rec_on_lfm, rec_on_svd, pr_lfm, pr_svd, title

    def plot_bar_chart(self, ndcg):
        x_range = np.arange(ndcg.shape[0])
        width = 0.25

        plt.figure(figsize=(6.4, 3.6))
        plt.title("NDCG Scores For Various Federation Techniques")
        plt.xlabel("Federation Technique")
        plt.ylabel("NDCG Score")
        plt.xticks(ticks=x_range, labels=ndcg[:, 4], rotation=90)
        plt.bar(x_range, ndcg[:, 0].astype(float), width=width,  label="LFM")
        plt.bar(x_range + width, ndcg[:, 1].astype(float), width=width, label="SVD")
        plt.legend()
        plt.savefig("DASD_federation_techniques.pdf", format="pdf", bbox_inches='tight')
        plt.show()

    def federate_results(self, n, reverse_mapping=False):
        # TODO: check performance difference between mapping svd to lfm AND lfm to svd

        self.log.info("Federating results (Mapping)...")

        # Get LFM's recs
        alg_warp = LightFMAlg("warp", ds=self.dataset, labels_ds=self.labels_ds, normalisation=self.norm_func)
        lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id, num_rec=-1)
        lfm_recs = np.c_[lfm_recs, np.full(lfm_recs.shape[0], "lfm")]  # append new column of "lfm" to recs

        # Get Surprise's SVD recs
        svd_split_filename = "/svd_split.npy".format()
        svd = SurpriseSVD(ds=self.dataset, normalisation=self.norm_func, save_filename=svd_split_filename,
                          load_filename=svd_split_filename)
        svd_recs = svd.get_top_n(self.user_id, n=-1)
        svd_recs = np.c_[svd_recs, np.full(svd_recs.shape[0], "svd")]  # append new column of "svd" to recs

        # Scale scores
        lfm_recs[:, 2] = helpers.scale_scores(lfm_recs[:, 2]).reshape(-1)
        svd_recs[:, 2] = helpers.scale_scores(svd_recs[:, 2]).reshape(-1)

        # Copy scores but don't map them (reshape to convert back to 1d row array)
        lfm_recs_unmapped = copy.deepcopy(lfm_recs)
        svd_recs_unmapped = copy.deepcopy(svd_recs)

        # Map one alg's score to another, then renormalise (reshape to convert back to 1d row array)
        if not reverse_mapping:
            svd_mapped = self.model.predict(svd_recs[:, 2].reshape(-1, 1).astype(float))
            b = helpers.scale_scores(svd_mapped).reshape(-1)
        else:
            lfm_mapped = self.model.predict(lfm_recs[:, 2].reshape(-1, 1).astype(float))
            lfm_recs[:, 2] = helpers.scale_scores(lfm_mapped).reshape(-1)

        # Get the maximum possible dcg score, if the rankings were to be perfectly ranked
        self.best_dcg_score = helpers.best_dcg_score(n)

        # Run all merging algorithms
        federated_recs, *merge_ndgc_tuple = self.merge_scores_by_mapping(lfm_recs, svd_recs, n=n)
        ndcg_tuples = np.array([merge_ndgc_tuple,
                                self.weave_scores_before_mapping(lfm_recs_unmapped, svd_recs_unmapped, n=n),
                                self.weave_scores_after_mapping(lfm_recs, svd_recs, n=n),
                                self.merge_by_raw_scores(lfm_recs_unmapped, svd_recs_unmapped, n=n),
                                self.alg_on_alg(lfm_recs_unmapped, n=n, title="Only LFM"),
                                self.alg_on_alg(svd_recs_unmapped, n=n, title="Only SVD"),
                                self.pick_random_baseline(federated_recs, n=n),
                                self.replace_random_baseline(federated_recs, n=n)
                                ])

        self.plot_bar_chart(ndcg_tuples)

        print("End")
        # TODO: Implement precision and recall and perhaps accuracy scores (would this work/tell me anything)


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    norm_func = helpers.gaussian_normalisation
    ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
    dh = DataHandler(filename=ds_path)

    # Filter out users and items below threshold
    dh, surviving_users = helpers.remove_below_threshold_user_and_items(dh, u_thresh=0, i_thresh=0)

    # Get users who have at least rated at least min_ratings movies
    min_ratings_users = helpers.get_users_with_min_ratings(surviving_users, min_ratings=10)
    user_id = np.min(min_ratings_users.index.astype(int))
    user_id = 1
    fed = FederatorMapper(user_id, data_path=dh.get_dataset(), labels_ds="/datasets/ml-latest-small/movies.csv",
                    norm_func=norm_func)
    fed.federate_results(50, reverse_mapping=False)
