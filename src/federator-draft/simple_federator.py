import numpy as np
import pandas as pd
from definitions import ROOT_DIR
import logging.config

from data_handler import DataHandler
from knn import KNearestNeighbours

DS_PATH = ROOT_DIR + "/datasets/ml-latest-small"


class SimpleFederator:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, name, n, base_n):
        ds_base_path = "/datasets/ml-latest-small"
        ds_path = ROOT_DIR + ds_base_path + "/ratings.csv"
        movie_id_col = 1

        self.log.info("Golden List:")
        base_list = KNearestNeighbours(ds_base_path, p_thresh=25, u_thresh=25)
        golden_list = base_list.make_recommendation(name, base_n, verbose=True)

        data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        num_of_alg_subsets = 5
        split_datasets = data.split_dataset_intermittently(num_of_alg_subsets)
        split_recommendations = self.knn_split_datasets(split_datasets, name, n, ds_base_path)
        federated_recommendations = self.federate_split_recommendations(split_recommendations, n)
        self.pretty_print_results(federated_recommendations, name)
        self.log.info("Score: %.3f" % self.measure_effectiveness(federated_recommendations, n, golden_list, base_n))

    # TODO: put in a different helper file
    def knn_split_datasets(self, split_datasets, name, n, ds_base_path):
        split_recommendations = []
        for i in range(len(split_datasets)):
            knn = KNearestNeighbours(ds_base_path, ds_ratings=self.convert_np_to_pandas(split_datasets[i]),
                                     p_thresh=0, u_thresh=0)
            split_recommendations.append(knn.make_recommendation(name, n))
        a = np.concatenate(split_recommendations)
        return a[a[:, 2].argsort()]  # sort the array by distance

    @staticmethod
    def convert_np_to_pandas(a, first_col=0, last_col=3):
        return pd.DataFrame(data=a[:, first_col:last_col], columns=['userId', 'movieId', 'rating'])

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

    def pretty_print_results(self, results, movie_name):
        self.log.info('Federated Recommendations for {}:'.format(movie_name))
        for row in results:
            self.log.info('{0}: {1}, with distance of {2}'.format(row[0], row[1], row[2]))

    def measure_effectiveness(self, actual, actual_n, golden, golden_n):
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


# Test with Pulp Fiction
if __name__ == '__main__':
    my_movie = "Pulp Fiction"
    n_neighbours = 20
    base_n_neighbours = 200

    federator = SimpleFederator(my_movie, n_neighbours, base_n_neighbours)
