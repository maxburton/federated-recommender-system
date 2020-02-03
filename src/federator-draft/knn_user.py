import numpy as np
from definitions import ROOT_DIR
import logging.config

import helpers
from knn import KNearestNeighbours


class KNNUser:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, data_path="/datasets/ml-latest-small", ds_ratings=None, min_rating=4.0, p_thresh=5,
                 u_thresh=5):

        knn = KNearestNeighbours(data_path, ds_ratings=ds_ratings, p_thresh=p_thresh, u_thresh=u_thresh)
        df_r = knn.df_ratings
        df_m = knn.df_movies
        liked_movies = df_r.loc[df_r['userId'] == user_id]
        liked_movies = liked_movies.loc[liked_movies['rating'] >= min_rating]
        liked_movies = df_m[df_m['movieId'].isin(liked_movies["movieId"])]['title']

        self.log.info("Liked Movies:")
        self.log.info(liked_movies.to_numpy())
        self.knn = knn
        self.liked_movies = liked_movies.to_numpy()
        self.user_id = user_id

    def make_recommendation(self, num_of_recs, verbose=False):
        unordered_recs = []
        for movie in self.liked_movies:
            sub_results = self.knn.make_recommendation(movie, n_recommendations=100, verbose=verbose)
            for rec in sub_results:
                unordered_recs.append(rec)
        unordered_recs = np.array(unordered_recs)
        ordered_recs = unordered_recs[unordered_recs[:, 2].argsort()]  # sort the array by distance

        final_recs_names = []
        final_recs = []
        rec_i = 0
        current_i = 0
        while rec_i < num_of_recs:
            _, movie_title, score = ordered_recs[current_i]
            if movie_title not in final_recs_names and movie_title not in self.liked_movies and float(score) >= 0.01:
                final_recs_names.append(movie_title)
                final_recs.append([rec_i + 1, movie_title, score])
                rec_i += 1
            current_i += 1
        helpers.pretty_print_results(self.log, final_recs, self.user_id)
        return np.array(final_recs)


if __name__ == '__main__':
    user_id = 1
    knnu = KNNUser(user_id, p_thresh=5, u_thresh=5)
    knnu.make_recommendation(num_of_recs=20)
