from sklearn.metrics import dcg_score
from sklearn.preprocessing import MinMaxScaler

from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits
import math
from data_handler import DataHandler
import numpy as np
import matplotlib.pyplot as plt
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

        self.complete_dataset = None
        self.size_ratios = None
        self.density_ratios = None
        self.user_activity_ratios = None
        self.total_ratings = None
        self.num_items_per_split = None

        # Get golden list for splits method
        self.golden_lfm_splits, self.golden_svd_splits = GoldenList().generate_lists(self.user_id,
                                                                                     data_path=self.data_path,
                                                                                     labels_ds=labels_ds,
                                                                                     num_of_recs=-1,
                                                                                     norm_func=norm_func)

        # Normalise golden list scores
        # self.golden_lfm_splits[:, 2] = helpers.scale_scores(self.golden_lfm_splits[:, 2]).flatten()
        # self.golden_svd_splits[:, 2] = helpers.scale_scores(self.golden_svd_splits[:, 2]).flatten()

        if self.alg == "lfm":
            self.golden = self.golden_lfm_splits
        else:
            self.golden = self.golden_svd_splits

        self.best_dcg_score = np.inf

    def calculate_metrics(self, federated_recs, n=10, title=""):
        federated_recs = helpers.remove_duplicate_recs(federated_recs)

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

    def plot_comparison(self, metrics, splitting_methods, titles):
        fig = plt.figure(figsize=(3.6, 3.6))
        ax = plt.subplot(111)
        plt.title("NDCG@k Scores For Various Splitting Methods")
        plt.ylabel("NDCG@k")
        plt.xlabel("Splitting Method")

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        length = metrics.shape[1]
        markers = [".", "o", ".", "o", "d", "s", "^"]
        k = np.arange(metrics.shape[0])
        plt.xticks(ticks=k, labels=splitting_methods, rotation=90)
        for i in range(length):
            ax.scatter(k, metrics[:, i], label=titles[i], marker=markers[i])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, shadow=True, ncol=4)
        save_filename = "SADD_comparison.pdf"
        fig.savefig(save_filename, format="pdf", bbox_inches='tight')
        plt.show()

    def plot_ndcg_against_num_recs(self, titles, metrics, min_k=5, splitting_method="even", verbose=False):
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("NDCG@k Scores For Various Federation Techniques & k")
        plt.ylabel("NDCG@k")
        plt.xlabel("k")

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])
        length = metrics.shape[0]
        markers = [".", "o", ".", "o", "d", "s", "^"]
        ls = [":", ":", "--", "--", "--", "-", "-"]

        for i in range(metrics.shape[1]):
            lw = 1.5

            k = np.arange(min_k, min_k + length)
            ax.plot(k, metrics[:, i], label=titles[i], ls=ls[i], lw=lw, marker=markers[i])
            if verbose:
                print(titles[i])
                for j in metrics[:, i]:
                    print(j)
                print("AVG: %.5f" % np.mean(metrics[:, i]))
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        save_filename = "SADD_ndcgk_scores_" + splitting_method + ".pdf"
        fig.savefig(save_filename, format="pdf", bbox_inches='tight')
        plt.show()

    def test_ecors_b(self, splits, n=10):
        ndcg_scores_b = []
        for i in range(0, 11):
            b = i / 10.0
            title = "ECoRS b=%.1f" % b
            metrics_tuple = self.ecors(splits, n=n, b=b, title=title)
            ndcg_scores_b.append(metrics_tuple[1][0])
        return ndcg_scores_b

    def plot_ecors_b(self, metrics, min_k=5, splitting_method="even"):
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("NDCG@k Scores For ECoRS while varying belief")
        plt.ylabel("NDCG@k")
        plt.xlabel("b")

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])

        for i in range(metrics.shape[1]):
            b = np.arange(0, 11) / 10.0
            k = min_k + i
            label = "k=%d" % k
            # only plot lines that have variance, i.e. I hasn't affected too much
            if np.max(metrics[i]) - np.min(metrics[i]) > 0.05:
                ax.plot(b, metrics[i], label=label, ls='--')
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
        save_filename = "SADD_ecors_b_scores_" + splitting_method + ".pdf"
        fig.savefig(save_filename, format="pdf", bbox_inches='tight')

    def plot_precision(self, titles, metrics):
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("Precision@k Scores For Various Federation Techniques")
        plt.ylabel("Precision@k")
        plt.xlabel("k")

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])
        length = metrics.shape[0]

        for i in range(length):
            precision_scores = metrics[i][0]
            k = np.arange(1, precision_scores.shape[0] + 1)
            ax.plot(k, precision_scores, label=titles[i], ls='--')
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        fig.savefig("SADD_precisionk_scores.pdf", format="pdf", bbox_inches='tight')
        plt.show()

    def run_on_splits(self, splitting_method=None, extract_pc=0, n_subsets=5, n=10, norm_func=None):
        isplits = IndividualSplits(n_subsets=n_subsets, extract_pc=extract_pc, user_id=self.user_id,
                                   splitting_method=splitting_method)
        self.complete_dataset = isplits.complete_dataset
        self.size_ratios = isplits.ratios
        self.density_ratios = isplits.densities
        self.user_activity_ratios = isplits.user_activity
        self.total_ratings = isplits.total_ratings
        self.num_items_per_split = isplits.num_items_per_split

        if self.alg == "lfm":
            splits = isplits.run_on_splits_lfm(self.user_id, num_of_recs=n, norm_func=norm_func)
        else:
            splits = isplits.run_on_splits_svd(self.user_id, norm_func=norm_func)

        return splits

    def weave(self, splits, n=10):
        len_splits = splits.shape[0]

        sorted_indices = []
        for i in range(len_splits):
            # Append split index to its max score
            sorted_indices.append([i, np.max(splits[i][:, 2].astype(float))])
        # sort in descending order of score
        sorted_indices = np.array(sorted_indices)
        sorted_indices = sorted_indices[np.argsort(sorted_indices[:, 1])][::-1]

        weave_recs = []
        i = 0
        while len(weave_recs) <= n:
            for j in range(len_splits):
                # Append top items from each split
                weave_recs.append(splits[sorted_indices[j][0].astype(int)][i])
            weave_recs = helpers.remove_duplicate_recs(weave_recs).tolist()
            i += 1

        metrics = self.calculate_metrics(weave_recs, title="Weave", n=n)

        return metrics

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

    def ecors(self, splits, n=10, threshold=0.005, b=0.4, title="ECoRS"):
        minmax = MinMaxScaler(feature_range=(0.5, 1))
        len_splits = splits.shape[0]
        t = np.array(self.num_items_per_split)
        # Get the scaling factor, which is larger when more relevant items exist
        num_complete_ds_items = np.unique(self.complete_dataset[:, 1]).shape[0]
        df_base = threshold * num_complete_ds_items * 0.001
        df_factor = df_base * 3

        dsat = 0
        at = []
        for i in range(len_splits):
            # num of items above threshold
            at_i = 0
            for movie in splits[i][:, 1]:
                at_i += self.rank_scorer(movie, ratio=threshold)
            at.append(at_i)

            # if a split contains at least n//3 items above threshold
            if at_i > (n//3):
                dsat += 1

        try:
            I = math.log((len_splits + 0.5) / dsat) / math.log(len_splits + 1.0) + 0.5
        # If no splits satisty dsat (dsat = 0)
        except ZeroDivisionError:
            I = 0
        T = []
        ecors = []
        for i in range(len_splits):
            # Normalising function
            NF = t[i] / np.sum(t) / 5
            avg_t = np.sum(t * NF) / len_splits

            # Calculate term frequency, T
            T.append(at[i] * NF / (at[i] * NF + (df_base * NF) + (df_factor * NF * t[i] * NF) / (avg_t * NF)))

        #T = minmax.fit_transform(np.array(T).reshape(-1, 1))
        for i in range(len_splits):
            ecors.append(b + (1 - b) * T[i] * I)
            splits[i][:, 2] = splits[i][:, 2].astype(float) * ecors[i]

        ecors_weighted = np.vstack(tuple(splits))
        ecors_weighted = helpers.sort_and_reset_rankings(ecors_weighted)

        metrics = self.calculate_metrics(ecors_weighted, title=title, n=n)

        return metrics

    def rors(self, splits, n=10, title="RoRS"):
        t = np.array(self.num_items_per_split)

        for i in range(splits.shape[0]):
            relevance_score = 0
            unique_movies = np.unique(splits[i][:, 1])
            for movie in unique_movies:
                NF = t[i] / np.sum(t)
                relevance_score += self.rank_scorer(movie) * NF
            splits[i][:, 2] = splits[i][:, 2].astype(float) * relevance_score

        rors_weighted = np.vstack(tuple(splits))
        rors_weighted = helpers.sort_and_reset_rankings(rors_weighted)

        metrics = self.calculate_metrics(rors_weighted, title=title, n=n)

        return metrics

    def rank_scorer(self, movie_title, ratio=0.005, positive_score=1):
        num_complete_ds_items = np.unique(self.complete_dataset[:, 1]).shape[0]
        try:
            golden_rank = int(self.golden[self.golden[:, 1] == movie_title][0][0])
        except IndexError:
            self.log.debug("Item not in golden list")
            return 0
        if golden_rank < (ratio * num_complete_ds_items):
            return positive_score
        return 0

    def generate_largevsmall_ratios(self, n_subsets=5, big_total=0.6):
        num_of_large = n_subsets // 5 + 1
        num_of_small = n_subsets - num_of_large

        big_ratios = np.random.dirichlet(np.ones(num_of_large), size=1).flatten() * big_total
        small_ratios = np.random.dirichlet(np.ones(num_of_small), size=1).flatten() * (1 - big_total)

        return np.concatenate((big_ratios, small_ratios))

    def federate(self, splitting_method=None, extract_pc=0, min_subsets=5, max_subsets=5, n=10):
        self.best_dcg_score = helpers.best_dcg_score(k=n)

        extract_pc_i = 0
        titles = []
        ndcg_scores = []
        pr_scores = []

        ecors_b = []

        for i in range(min_subsets, max_subsets+1):
            splitting_method_i = splitting_method
            if splitting_method == "lvs":
                splitting_method_i = self.generate_largevsmall_ratios(n_subsets=i)
            elif splitting_method == "dvs":
                splitting_method_i = self.generate_largevsmall_ratios(n_subsets=i-1)
                extract_pc_i = extract_pc
            splits = self.run_on_splits(n_subsets=i, n=n, splitting_method=splitting_method_i, extract_pc=extract_pc_i,
                                        norm_func=norm_func)
            self.calculate_metrics_on_splits(splits, n=n)

            metric_tuples = [self.raw_merge(np.copy(splits), n=n),
                             self.weave(np.copy(splits), n=n),
                             self.db_size(np.copy(splits), n=n),
                             self.db_density(np.copy(splits), n=n),
                             self.user_activity(np.copy(splits), n=n),
                             self.ecors(np.copy(splits), n=n),
                             self.rors(np.copy(splits), n=n),
                             ]

            titles = np.array([metric_tuples[i][0] for i in range(len(metric_tuples))])
            ndcg_scores.append([metric_tuples[i][1][0] for i in range(len(metric_tuples))])
            pr_scores.append([metric_tuples[i][1][1] for i in range(len(metric_tuples))])
            # precisions_scores = np.array([metric_tuples[i][2] for i in range(len(metric_tuples))])

            ecors_b.append(self.test_ecors_b(np.copy(splits), n=n))

        self.plot_ndcg_against_num_recs(titles, np.array(ndcg_scores), min_k=min_subsets, splitting_method=splitting_method)
        self.ndcg_scores = ndcg_scores
        self.titles = titles

        print("done")


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    norm_func = None
    ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
    dh = DataHandler(filename=ds_path)

    # Filter out users and items below threshold
    dh, surviving_users, _ = helpers.remove_below_threshold_user_and_items(dh.dataset, u_thresh=0, i_thresh=0)

    # Get users who have at least rated at least min_ratings movies
    min_ratings_users = helpers.get_users_with_min_ratings(surviving_users, min_ratings=10)
    user_id = np.min(min_ratings_users.index.astype(int))

    # manually set user id
    user_id = 5
    fed = FederatorSplits(user_id, alg="lfm", data_path=dh.get_dataset(),
                          labels_ds="/datasets/ml-latest-small/movies.csv", norm_func=norm_func)
    # splitting_method = "random"
    # splitting_method = [0.5, 0.3, 0.1, 0.05, 0.05]
    splitting_method = "lvs"
    extract_pc = 10
    fed.federate(splitting_method=splitting_method, extract_pc=extract_pc, max_subsets=20, n=20)

    #plot comparison
    methods = ["even", "random", "lvs", "dvs"]
    ndcg_scores = []
    for method in methods:
        fed.federate(splitting_method=method, extract_pc=extract_pc, min_subsets=20, max_subsets=20, n=20)
        ndcg_scores.append(np.array(fed.ndcg_scores).flatten())
    fed.plot_comparison(np.array(ndcg_scores), np.array(methods), fed.titles)

