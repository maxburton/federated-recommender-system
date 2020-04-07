import logging.config
import multiprocessing

import numpy as np

import helpers
from data_handler import DataHandler
from definitions import ROOT_DIR
from lightfm_alg import LightFMAlg
from surprise_svd import SurpriseSVD

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, SGDRegressor


class AlgMapper:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, n_subsets=5, movie_id_col=1, data_path=None, labels_ds=None, split_to_train=0, norm_func=None):
        if data_path is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
            data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        elif type(data_path) is str:
            ds_path = ROOT_DIR + data_path
            data = DataHandler(filename=ds_path)
        else:
            data = DataHandler(ds=data_path)

        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        split_dataset = data.split_dataset_intermittently(n_subsets)
        split_data = split_dataset[split_to_train]
        self.untrained_data = split_dataset[np.arange(split_dataset.shape[0]) != split_to_train]  # all but one index

        """
        knnu = KNNUser(user_id, ds_ratings=helpers.convert_np_to_pandas(split_data),
                       p_thresh=0, u_thresh=0)
        self.knn_recs = knnu.make_recommendation(num_of_recs=-1)  # -1 means get all available
        """

        alg_warp = LightFMAlg("warp", ds=split_data, labels_ds=labels_ds, normalisation=norm_func)  # warp or bpr
        self.lfm_recs = alg_warp.generate_rec(alg_warp.model, user_id, num_rec=-1)

        svd_train_split_filename = "/svd_mapper_{0}.npy".format(split_to_train)
        svd = SurpriseSVD(ds=split_data, normalisation=norm_func, save_filename=svd_train_split_filename, load_filename=svd_train_split_filename)
        self.svd_recs = svd.get_top_n(user_id, n=-1)

    def remove_duplicates(self, array, col):
        unique_array = []
        unique_titles = []
        for row in array:
            if row[col] not in unique_titles:
                unique_array.append(row)
                unique_titles.append(row[col])
        return np.array(unique_array)

    # normalise the data to range 0-1, after removing unique (to one algorithm) entries and sorting by movie title
    def normalise_and_trim(self):
        # Remove duplicates
        lfm_unique = self.remove_duplicates(self.lfm_recs, 1)
        svd_unique = self.remove_duplicates(self.svd_recs, 1)

        # Sort all entries by title so their indexes are synced
        lfm_sorted = lfm_unique[lfm_unique[:, 1].argsort()]
        svd_sorted = svd_unique[svd_unique[:, 1].argsort()]

        # Remove all entries that don't exist in both lists
        lfm_mask = np.in1d(lfm_sorted[:, 1], svd_sorted[:, 1])
        svd_mask = np.in1d(svd_sorted[:, 1], lfm_sorted[:, 1])
        lfm_sorted = lfm_sorted[lfm_mask]
        svd_sorted = svd_sorted[svd_mask]

        lfm_normalised_scores = helpers.scale_scores(lfm_sorted[:, 2])  # reshape to a 1d array represented as a column
        svd_normalised_scores = helpers.scale_scores(svd_sorted[:, 2])
        return lfm_normalised_scores, svd_normalised_scores

    def learn_mapping(self, scores1, scores2):
        x_train, x_test, y_train, y_test = train_test_split(scores1, scores2, test_size=0.2)

        pipe = Pipeline([
            ('regr', Lasso())
        ])

        param_grid = [
            {
                'regr': [Lasso(), Ridge()],
                'regr__alpha': np.logspace(-4, 1, 6),
            },
            {
                'regr': [SGDRegressor()],
                'regr__alpha': np.logspace(-5, 0, 6),
                'regr__max_iter': [500, 1000],
            },
        ]

        grid = GridSearchCV(pipe, scoring="neg_root_mean_squared_error", param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)  # If crash, change to n_jobs=1
        grid.fit(x_train, y_train)

        #predicted = grid.predict(x_test)

        self.log.info('Score:\t{}'.format(grid.score(x_test, y_test)))
        return grid


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    user_id = 1
    mapper = AlgMapper(user_id, split_to_train=0)
    svd, lfm = mapper.normalise_and_trim()
    model = mapper.learn_mapping(svd, lfm)
    helpers.create_scatter_graph("Normalised SVD vs LFM scores", "Movie IDs", "Normalised Score", ["SVD", "LFM"],
                                 ["red", "blue"], svd, lfm)
