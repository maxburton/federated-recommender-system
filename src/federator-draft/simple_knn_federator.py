import numpy as np
import pandas as pd
from definitions import ROOT_DIR
import logging.config

import helpers
from data_handler import DataHandler
from knn_user import KNNUser

DS_PATH = ROOT_DIR + "/datasets/ml-latest-small"


class SimpleKNNFederator:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, n, base_n):
        ds_base_path = "/datasets/ml-latest-small"
        ds_path = ROOT_DIR + ds_base_path + "/ratings.csv"
        movie_id_col = 1

        self.log.info("Golden List:")
        num_of_recs = 20
        golden_list = KNNUser(user_id, data_path=ds_base_path, p_thresh=5, u_thresh=5).make_recommendation(num_of_recs)

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        num_of_alg_subsets = 5
        split_datasets = data.split_dataset_intermittently(num_of_alg_subsets)
        split_recommendations = self.knn_split_datasets(split_datasets, user_id, num_of_recs)
        federated_recommendations = self.federate_split_recommendations(split_recommendations, n)
        helpers.pretty_print_results(self.log, federated_recommendations, user_id)
        self.log.info("Score: %.3f" % self.measure_effectiveness(federated_recommendations, n, golden_list, base_n))

    # TODO: put in a different helper file
    def knn_split_datasets(self, split_datasets, user_id, num_of_recs=20):
        split_recommendations = []
        for i in range(len(split_datasets)):
            knnu = KNNUser(user_id, ds_ratings=helpers.convert_np_to_pandas(split_datasets[i]),
                           p_thresh=0, u_thresh=0)
            split_recommendations.append(knnu.make_recommendation(num_of_recs=num_of_recs))
        a = np.concatenate(split_recommendations)
        return a[a[:, 2].argsort()]  # sort the array by distance

    @staticmethod
    def federate_split_recommendations(recommendations, n):
        federated_recommendations = []
        chosen_names = []
        entries = 0
        count = 0
        while entries < n:
            movie = recommendations[count]
            name = movie[1]
            dist = movie[2]
            if name not in chosen_names:
                chosen_names.append(name)
                federated_recommendations.append([entries+1, name, dist])
                entries += 1
            count += 1
        return federated_recommendations

    def measure_effectiveness(self, actual, actual_n, golden, golden_n):  # TODO: Replace with a real metric measurer
        golden_dict = {}
        score = 1.0

        for row in golden:
            golden_dict[row[1]] = row[0]

        for row in actual:
            name = row[1]
            position = row[0]
            if name in golden_dict:
                score -= (1/actual_n)*((abs(golden_dict[name] - position))/golden_n)
            else:
                score -= (1/actual_n)
        return score


# Test with user 1
if __name__ == '__main__':
    user_id = 1
    n_neighbours = 20
    base_n_neighbours = 200

    federator = SimpleKNNFederator(user_id, n_neighbours, base_n_neighbours)
