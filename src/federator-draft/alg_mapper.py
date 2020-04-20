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
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import matplotlib.pyplot as plt


class AlgMapper:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, n_subsets=5, movie_id_col=1, data_path=None, labels_ds=None, split_to_train=0,
                 norm_func=None, use_full_dataset=False):

        self.log.info("Mapping algs...")

        if data_path is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
            data = DataHandler(filename=ds_path, dtype=np.uint32, cols=4)
        elif type(data_path) is str:
            ds_path = ROOT_DIR + data_path
            data = DataHandler(filename=ds_path)
        else:
            data = DataHandler(ds=data_path)

        data.set_dataset(data.sort_dataset_by_col(movie_id_col))
        if not use_full_dataset:
            split_dataset = data.split_dataset_intermittently(n_subsets)
            split_data = split_dataset[split_to_train]
            self.untrained_data = split_dataset[np.arange(split_dataset.shape[0]) != split_to_train]  # all but one index
        else:
            split_data = data.get_dataset()
            self.untrained_data = data.get_dataset()

        lfm_train_split_filename = "/lfm_mapper_{0}.npy".format(split_to_train)
        alg_warp = LightFMAlg("warp", ds=split_data, labels_ds=labels_ds, normalisation=norm_func,
                              sl_filename=lfm_train_split_filename)
        self.lfm_recs = alg_warp.generate_rec(user_id, num_rec=-1)

        svd_train_split_filename = "/svd_mapper_{0}.npy".format(split_to_train)
        svd = SurpriseSVD(ds=split_data, normalisation=norm_func, sl_filename=svd_train_split_filename)
        self.svd_recs = svd.get_top_n(user_id, n=-1)

        if norm_func is not None:
            self.normalised_ratings = (svd.unnormalised_ratings, svd.normalised_ratings)

    def plot_mse(self, gbr, lr, n_item_range):
        fig = plt.figure()
        fig.set_size_inches(6.4, 2.8)
        ax = plt.subplot(111)
        plt.title("MSE Effectiveness on Items Trained")
        plt.ylabel("MSE")
        plt.xlabel("Number of Items Trained")

        ax.plot(n_item_range, gbr, label="gbr")
        ax.plot(n_item_range, lr, label="lr")
        ax.legend()
        # Put a legend below current axis
        save_filename = "DASD_mse_scores.pdf"
        fig.savefig(save_filename, format="pdf", bbox_inches='tight')
        fig.show()

    def plot_boxplot(self, mse, title="Difference in Training Data", ylabel="xtrain - ytrain"):
        fig = plt.figure(figsize=(3.6, 3.6))
        ax = plt.subplot(111)
        plt.title(title)
        plt.ylabel(ylabel)

        ax.boxplot(mse, notch=True)
        ax.set_xticklabels([])
        # Put a legend below current axis
        save_filename = "DASD_diff_boxplot.pdf"
        fig.savefig(save_filename, format="pdf", bbox_inches='tight')
        fig.show()

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

        lfm_normalised_scores = helpers.scale_scores(lfm_sorted[:, 2]).astype(float).reshape(-1, 1)
        svd_normalised_scores = helpers.scale_scores(svd_sorted[:, 2]).astype(float).reshape(-1, 1)
        return lfm_normalised_scores, svd_normalised_scores

    def trim_to_item_cap(self, item_cap, x, y):
        try:
            return x[:item_cap], y[:item_cap]
        except IndexError:
            return x, y

    def learn_mapping(self, scores1, scores2, item_cap=None):
        mapping_func = self.learn_mapping_gbr
        return mapping_func(scores1, scores2, item_cap=item_cap)

    def learn_mapping_gbr(self, scores1, scores2, item_cap=None, cv=False, plot=False):
        x_train, x_test, y_train, y_test = train_test_split(scores1, scores2, test_size=0.2)
        if item_cap:
            x_train, y_train = self.trim_to_item_cap(item_cap, x_train, y_train)

        # plot difference boxplot
        if plot:
            all_diff = x_train - y_train
            self.plot_boxplot(all_diff)

        if cv:
            pipe = Pipeline([
                ('gbr', ensemble.GradientBoostingRegressor())
            ])

            param_grid = [
                {
                    'gbr__n_estimators': [500],
                    'gbr__max_depth': [3, 5, 1000],
                    'gbr__learning_rate': np.logspace(-3, -1, 3),
                    'gbr__loss': ['ls']
                }
            ]

            # If crash, change to n_jobs=1
            clf = GridSearchCV(pipe, scoring="neg_root_mean_squared_error", param_grid=param_grid, cv=3, n_jobs=-1,
                               verbose=1)
            clf.fit(x_train, y_train.ravel())
        else:
            params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}
            clf = ensemble.GradientBoostingRegressor(**params)
            clf.fit(x_train, y_train.ravel())

        predicted = clf.predict(x_test)
        self.log.debug(predicted)

        mse = mean_squared_error(y_test, clf.predict(x_test))
        print("GBR MSE: %.4f" % mse)
        return clf, mse

    def learn_mapping_linear(self, scores1, scores2, item_cap=None, cv=True):
        x_train, x_test, y_train, y_test = train_test_split(scores1, scores2, test_size=0.2)
        if item_cap:
            x_train, y_train = self.trim_to_item_cap(item_cap, x_train, y_train)

        if cv:
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

            # If crash, change to n_jobs=1
            lr = GridSearchCV(pipe, scoring="neg_root_mean_squared_error", param_grid=param_grid, cv=5, n_jobs=-1)
            lr.fit(x_train, y_train.ravel())
        else:
            params = {'max_iter': 500, 'alpha': 0.1}
            lr = Lasso(**params)
            lr.fit(x_train, y_train.ravel())

        mse = mean_squared_error(y_test, lr.predict(x_test))
        print('LR MSE: %.4f' % mse)
        return lr, mse


if __name__ == '__main__':
    # Allows n_jobs to be > 1
    multiprocessing.set_start_method('spawn')

    user_id = 5
    norm = None
    mapper = AlgMapper(user_id, split_to_train=0, norm_func=norm)
    lfm, svd = mapper.normalise_and_trim()
    helpers.create_scatter_graph("SVD vs LFM scores (Decouple)", "Movie IDs", "Normalised Score", ["LFM", "SVD"],
                                 ["blue", "orange"], lfm, svd)

    model = mapper.learn_mapping_gbr(svd, lfm, cv=False)

    """
    n_item_range = np.arange(1, 500)
    mse_gbr = []
    for i in n_item_range:
        _, gbr = mapper.learn_mapping_gbr(svd, lfm, item_cap=i, cv=False)
        mse_gbr.append(gbr)
    """
