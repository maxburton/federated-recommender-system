import numpy as np
from definitions import ROOT_DIR
import logging.config

import helpers
from knn import KNearestNeighbours


class KNNUser:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, user_id, min_rating=4.0, num_of_recs=10):

        knn = KNearestNeighbours("/datasets/ml-latest-small", p_thresh=5, u_thresh=5)
        df_r = knn.df_ratings
        df_m = knn.df_movies
        liked_movies = df_r.loc[df_r['userId'] == user_id]
        liked_movies = liked_movies.loc[liked_movies['rating'] >= min_rating]
        liked_movies = df_m[df_m['movieId'].isin(liked_movies["movieId"])]['title']
        self.log.info("Liked movies: " + liked_movies)

        unordered_recs = []
        for movie in liked_movies:
            sub_results = knn.make_recommendation(movie, 10, verbose=False)
            for rec in sub_results:
                unordered_recs.append(rec)
        unordered_recs = np.array(unordered_recs)
        ordered_recs = unordered_recs[unordered_recs[:, 2].argsort()]  # sort the array by distance

        final_recs_names = []
        final_recs = []
        i = 0
        while i < num_of_recs:
            rec = ordered_recs[i]
            _, movie_title, score = rec
            if movie_title not in final_recs:
                final_recs_names.append(movie_title)
                final_recs.append([i+1, movie_title, score])
                i += 1
        helpers.pretty_print_results(self.log, final_recs, user_id)


if __name__ == '__main__':
    user_id = 1
    KNNUser(user_id, num_of_recs=10)

