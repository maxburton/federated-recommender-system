import os
import pandas as pd
from pivot_matrix import PivotMatrix
import fuzzymatcher as fm
from definitions import ROOT_DIR
import logging.config
from sklearn.neighbors import NearestNeighbors

DS_PATH = ROOT_DIR + "/datasets/ml-latest-small"


# TODO: Make this work with my data slicer
class KNearestNeighbours:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    df_movies = None
    df_ratings = None
    model_knn = None
    movie_features = None
    movie_features_sparse = None

    def __init__(self, p_thresh=50, u_thresh=50, m_filename="movies.csv", r_filename="ratings.csv"):
        # configure file path
        movies_filename = m_filename
        ratings_filename = r_filename

        # read data
        self.df_movies = pd.read_csv(
            os.path.join(DS_PATH, movies_filename),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})

        self.df_ratings = pd.read_csv(
            os.path.join(DS_PATH, ratings_filename),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        popularity_threshold = p_thresh
        popular_movies = self.df_ratings.groupby('movieId').filter(lambda x: len(x) > popularity_threshold)
        self.log.info('shape of original ratings data: %s' % str(self.df_ratings.shape))
        self.log.info('shape of ratings data after dropping unpopular movies: %s' % str(popular_movies.shape))

        ratings_threshold = u_thresh
        active_users = popular_movies.groupby('userId').filter(lambda x: len(x) > ratings_threshold)
        self.log.info('shape of ratings data after dropping both unpopular movies and inactive users: %s' %
                      str(active_users.shape))

        self.df_ratings = active_users

        # pivot ratings into a sparse movie x user matrix
        self.movie_features_sparse = PivotMatrix().pivot(self.df_ratings)

        # make an object for the NearestNeighbors Class.
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

        self.model_knn.fit(self.movie_features_sparse)

    def make_recommendation(self, input_movie, n_recommendations=10):
        t2id, id2csr = self.create_mappers(self.df_ratings, self.df_movies)
        self.log.info('You have input movie: %s' % input_movie)

        # get the movie's ID and map it to the sparse matrix row index
        idx = id2csr[fm.fuzzy_matching(t2id, input_movie)]

        self.log.info('KNN model predicting...')
        distances, indices = self.model_knn.kneighbors(self.movie_features_sparse[idx], n_neighbors=n_recommendations+1)

        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]

        # reverse each mapper to map backwards
        r_id2csr = {v: k for k, v in id2csr.items()}
        r_t2id = {v: k for k, v in t2id.items()}

        self.log.info('Recommendations for {}:'.format(input_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            self.log.info('{0}: {1}, with distance of {2}'.format(i + 1, r_t2id[r_id2csr[idx]], dist))

    @staticmethod
    def create_mappers(ratings, movies):
        surviving_movie_ids = sorted(ratings["movieId"].unique())
        df_movies_idx = movies.set_index(movies["movieId"].values)

        movie_title2movie_id = {
            movie: i for i, movie in
            zip(surviving_movie_ids, df_movies_idx.loc[surviving_movie_ids].values[:, 1])
        }

        movie_id2sparse_index = {
            movie_id: i for i, movie_id in
            enumerate(surviving_movie_ids)
        }
        return movie_title2movie_id, movie_id2sparse_index


if __name__ == '__main__':
    my_movie = "Pulp Fiction"

    knn = KNearestNeighbours()
    knn.make_recommendation(my_movie, 300)
