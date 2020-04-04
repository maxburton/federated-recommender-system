from surprise_svd import SurpriseSVD
from lightfm_alg import LightFMAlg
from data_handler import DataHandler
from definitions import ROOT_DIR
from helpers import svd_precision_recall_at_k, create_scatter_graph

import logging.config
import numpy as np
from lightfm.evaluation import precision_at_k


class TestAlgorithmMetrics:
    def __init__(self):
        logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
        log = logging.getLogger(__name__)

        movie_id_col = 1
        n_subsets = 3
        k = 10

        ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))

        lfm_results = []
        svd_results = []
        x = np.arange(n_subsets)

        for i in x:
            split_dataset = data.split_dataset_intermittently(i+1)

            lfm = LightFMAlg("warp", ds=split_dataset[0])
            all_user_precisions = precision_at_k(lfm.model, lfm.train, k=k)
            precision_lfm = sum(all_user_precisions) / len(all_user_precisions)
            lfm_results.append(precision_lfm)
            log.info("LFM: %.5f" % precision_lfm)

            svd = SurpriseSVD(ds=split_dataset[0])
            all_user_precisions, _ = svd_precision_recall_at_k(svd.predictions, k=k)
            precision_svd = sum(prec for prec in all_user_precisions.values()) / len(all_user_precisions)
            svd_results.append(precision_svd)
            log.info("SVD: %.5f" % precision_svd)

        title = "Precision@%d for LFM and SVD running on datasets of decreasing size" % k
        x_label = "Dataset size"
        y_label = "Precision@k Score"
        create_scatter_graph(title, x_label, y_label, ["LFM", "SVD"], ["red", "blue"], lfm_results, svd_results, x=x)

if __name__ == '__main__':
    TestAlgorithmMetrics()
