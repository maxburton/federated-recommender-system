from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
from data_handler import DataHandler
from definitions import ROOT_DIR
import helpers
import matplotlib.pyplot as plt

import logging.config
import numpy as np
from lightfm.evaluation import precision_at_k


class TestAlgorithmMetrics:
    def __init__(self, norm_func=None):
        logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
        log = logging.getLogger(__name__)

        movie_id_col = 1
        n_subsets = 10
        k = 10

        ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))

        lfm_results = []
        svd_results = []
        x = np.arange(n_subsets)
        labels = []
        width = 0.35

        for i in x:
            split_dataset = data.split_dataset_intermittently(i+1)
            labels.append(100 // (i+1))

            lfm = LightFMAlg("warp", ds=split_dataset[0], normalisation=norm_func)
            all_user_precisions = precision_at_k(lfm.model, lfm.train, k=k)
            precision_lfm = sum(all_user_precisions) / len(all_user_precisions)
            lfm_results.append(precision_lfm)
            log.info("LFM: %.5f" % precision_lfm)

            svd = SurpriseSVD(ds=split_dataset[0], normalisation=norm_func)
            all_user_precisions, _ = helpers.svd_precision_recall_at_k(svd.predictions, k=k)
            precision_svd = sum(prec for prec in all_user_precisions.values()) / len(all_user_precisions)
            svd_results.append(precision_svd)
            log.info("SVD: %.5f" % precision_svd)

        plt.title("Precision@{0} for LFM and SVD Running on Various Dataset Sizes".format(k))
        plt.xlabel("Dataset Size (in 1000s of Ratings)")
        plt.ylabel("Precision@{0} Score".format(k))
        plt.xticks(ticks=x, labels=labels)
        plt.bar(x, lfm_results, width=width, label="LFM")
        plt.bar(x + width, svd_results, width=width, label="SVD")
        plt.legend()
        plt.savefig("initial_precision.pdf", format("pdf"))
        plt.show()


if __name__ == '__main__':
    norm_func = None
    TestAlgorithmMetrics(norm_func)
