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

        self.final_recs_names = []
        self.final_recs = []
        self.rec_i = 0
        self.current_i = 0
        self.ordered_recs = []

    def make_recommendation(self, num_of_recs, n_recommendations=100, min_score=0.01, verbose=False):
        unordered_recs = []
        for movie in self.liked_movies:
            sub_results = self.knn.make_recommendation(movie, n_recommendations=n_recommendations, verbose=verbose)
            for rec in sub_results:
                unordered_recs.append(rec)
        unordered_recs = np.array(unordered_recs)
        self.ordered_recs = unordered_recs[unordered_recs[:, 2].argsort()]  # sort the array by distance

        if num_of_recs == -1:
            while True:
                try:
                    self.get_next_rec(min_score)
                except IndexError:
                    self.log.info("Maximum recs reached! Aborting loop")
                    break
        else:
            while self.rec_i < num_of_recs:
                self.get_next_rec(min_score)
        helpers.pretty_print_results(self.log, self.final_recs, self.user_id)
        return np.array(self.final_recs)

    def get_next_rec(self, min_score):
        _, movie_title, score = self.ordered_recs[self.current_i]
        if movie_title not in self.final_recs_names and \
                movie_title not in self.liked_movies and float(score) >= min_score:
            self.final_recs_names.append(movie_title)
            self.final_recs.append([self.rec_i + 1, movie_title, score])
            self.rec_i += 1
        self.current_i += 1


if __name__ == '__main__':
    user_id = 1
    knnu = KNNUser(user_id, p_thresh=5, u_thresh=5)
    knnu.make_recommendation(num_of_recs=20)
