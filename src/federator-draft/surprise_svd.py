from surprise import Dataset, Reader, SVD, dump
from definitions import ROOT_DIR
import logging.config
import pandas as pd
import numpy as np
import helpers
import time


class SurpriseSVD:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    # Can save and load the svd array to file
    def __init__(self, ds=None, normalisation=None, save=True, load=False, sl_filename="svd", base_folder="/svd_dumps/",
                 movies_filename="/datasets/ml-latest-small/movies.csv"):
        # Create mapper from movie id to title
        self.mid2title = helpers.generate_id2movietitle_mapper(filename=movies_filename)

        # Read data from file (or ds)
        if ds is None:
            df = pd.read_csv(ROOT_DIR+"/datasets/ml-latest-small/ratings.csv", usecols=[0, 1, 2])
        elif type(ds) is str:
            df = pd.read_csv(ROOT_DIR + ds, usecols=[0, 1, 2])
        else:
            df = pd.DataFrame(ds[:, 0:3], columns=["userId", "movieId", "rating"])

        lower = np.min(df['rating'].to_numpy())
        upper = np.max(df['rating'].to_numpy())

        # Normalise ratings
        if normalisation:
            self.unnormalised_ratings = np.copy(df['rating'].to_numpy())
            df['rating'] = normalisation(df[['userId', 'rating']].to_numpy())
            self.normalised_ratings = np.copy(df['rating'].to_numpy())
        reader = Reader(rating_scale=(lower, upper))
        self.data = Dataset.load_from_df(df, reader=reader)
        self.trainset = self.data.build_full_trainset()

        sl_filename = ROOT_DIR + base_folder + sl_filename

        start_time = time.time()
        self.log.info("Generating SVD model...")
        # Try to load existing SVD file from local storage
        if load:
            self.log.info("Attempting to load SVD alg from local storage...")
            try:
                _, self.alg = dump.load(sl_filename)
                self.log.info("SVD alg loaded!")
            except FileNotFoundError:
                self.log.info("File doesn't exist! Generating SVD from scratch.")
                self.alg = SVD(n_epochs=30, n_factors=40)
                self.alg.fit(self.trainset)
        else:
            self.alg = SVD(n_epochs=30, n_factors=40)
            self.alg.fit(self.trainset)

        # Save SVD alg to local storage
        if save:
            dump.dump(sl_filename, algo=self.alg)

        print("SVD train time: %.3f" % (time.time() - start_time))

    def print_user_favourites(self, user_id, min_rating=4.0):
        ratings = np.array(self.data.raw_ratings)
        mask = ratings[:, 0] == str(user_id)
        users_ratings = ratings[mask]
        positive_ratings = users_ratings[users_ratings[:, 2].astype(float) >= min_rating]
        self.log.info("User %d's favourite movies:" % user_id)
        for rating in positive_ratings:
            self.log.info("%s, %.1f" % (self.mid2title[int(rating[1])], float(rating[2])))

    def get_top_n(self, user_id, n=10, verbose=False):
        """Return the top-n recommendation for a user from a set of predictions.

        Args:
            user_id(int): The user we are generating the recommendations for
            n(int): The number of recommendation to output for the user. Default
                is 10. -1 returns all available recommendations.
            verbose(bool): if True, prints the top n results

        Returns:
        An np array, with recommendations in the form of [movieId, title, score]
        """

        # Create a testset for user_id that doesn't include existing ratings
        #testset = np.array(helpers.build_anti_testset_memory_managed(self.trainset, user_id))
        testset = np.array(self.trainset.build_anti_testset())
        testset = testset[testset[:, 0].astype(int) == user_id]

        # Estimate ratings for user_id
        start_time = time.time()
        self.predictions = np.array(self.alg.test(testset))
        print("SVD predict time: %.3f" % (time.time() - start_time))

        # Get user row and append all item scores
        top_n = []
        for uid, iid, true_r, est, _ in self.predictions:
            top_n.append((iid, est))

        if n == -1:
            n = len(top_n)

        # Then sort the predictions for the user and retrieve the n highest ones.
        top_n.sort(key=lambda x: x[1], reverse=True)
        top_n = top_n[:n]
        recs = []

        # Get the top n recommendations
        for i in range(n):
            movie_id = int(top_n[i][0])
            score = top_n[i][1]
            recs.append([i+1, self.mid2title[movie_id], score])

        if verbose:
            self.print_user_favourites(user_id)
            self.log.info("Recommendations for user %d" % user_id)
            helpers.pretty_print_results(self.log, recs, user_id + 1)

        return np.array(recs)


if __name__ == '__main__':
    user_id = 5

    svd = SurpriseSVD()
    results1 = svd.get_top_n(user_id, n=20)

    svd = SurpriseSVD()
    results2 = svd.get_top_n(user_id, n=20)

    print("end")
