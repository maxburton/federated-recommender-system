from surprise import Dataset, Reader, SVD
from surprise.model_selection.validation import cross_validate
from definitions import ROOT_DIR
import logging.config
from collections import defaultdict
import pandas as pd
import numpy as np


class SurpriseSVD:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, ds=None):
        df_movies = pd.read_csv(
            ROOT_DIR + "/datasets/ml-latest-small/movies.csv",
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int', 'title': 'str'})
        self.mid2title = {k: v for k, v in zip(df_movies["movieId"], df_movies["title"])}

        reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
        if ds is None:
            self.data = Dataset.load_from_file(ROOT_DIR+"/datasets/ml-latest-small/ratings.csv", reader=reader)
        else:
            reader = Reader(line_format='user item rating timestamp', sep=",")
            self.data = Dataset.load_from_df(pd.DataFrame(ds[:, 0:3], columns=["user", "item", "rating"]), reader=reader)
        algo = SVD()
        #results = cross_validate(algo, data, measures=['RMSE', 'MAE'])
        #print(repr(results))

        trainset = self.data.build_full_trainset()
        testset = trainset.build_anti_testset()

        algo.fit(trainset)
        self.predictions = np.array(algo.test(testset))  # TODO: See if I can make this faster (e.g. only calculate one user)

    def print_user_favourites(self, user_id, min_rating=4.0):
        ratings = np.array(self.data.raw_ratings)
        mask = ratings[:, 0] == str(user_id)
        users_ratings = ratings[mask]
        positive_ratings = users_ratings[users_ratings[:, 2].astype(float) >= min_rating]
        self.log.info("User %d's favourite movies:" % user_id)
        for rating in positive_ratings:
            self.log.info("%s, %.1f" % (self.mid2title[rating[1]], float(rating[2])))

    def get_top_n(self, user_id, n=10, verbose=True):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
            verbose(bool): if True, prints the top n results

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to the user.
        top_n = []

        mask = self.predictions[:, 0].astype(int) == user_id
        for uid, iid, true_r, est, _ in self.predictions[mask]:
            top_n.append((iid, est))

        if n == -1:
            n = len(top_n)

        # Then sort the predictions for the user and retrieve the n highest ones.
        top_n.sort(key=lambda x: x[1], reverse=True)
        top_n = top_n[:n]

        recs = []

        if verbose:
            self.log.info("Recommendations for user %d" % user_id)
        for i in range(n):
            movie_id = int(top_n[i][0])
            score = top_n[i][1]
            recs.append([i+1, self.mid2title[movie_id], score])
            if verbose:
                self.log.info("%d: %s - %.3f" % (i+1, self.mid2title[movie_id], score))

        return np.array(recs)


if __name__ == '__main__':
    user_id = 1
    svd = SurpriseSVD()
    svd.print_user_favourites(user_id)
    results = svd.get_top_n(user_id, n=20)
