import os
import pandas as pd
from scipy.sparse import csr_matrix
from definitions import ROOT_DIR
import logging.config
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

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

    def __init__(self):
        # configure file path
        movies_filename = 'movies.csv'  # Todo: remove these 2 variables for something more abstract
        ratings_filename = 'ratings.csv'

        # read data
        self.df_movies = pd.read_csv(
            os.path.join(DS_PATH, movies_filename),
            usecols=['movieId', 'title'],
            dtype={'movieId': 'int32', 'title': 'str'})

        self.df_ratings = pd.read_csv(
            os.path.join(DS_PATH, ratings_filename),
            usecols=['userId', 'movieId', 'rating'],
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        popularity_threshold = 50
        popular_movies = self.df_ratings.groupby('movieId').filter(lambda x: len(x) > popularity_threshold)
        self.log.info('shape of original ratings data: %s' % str(self.df_ratings.shape))
        self.log.info('shape of ratings data after dropping unpopular movies: %s' % str(popular_movies.shape))

        ratings_threshold = 50
        active_users = popular_movies.groupby('userId').filter(lambda x: len(x) > ratings_threshold)
        self.log.info('shape of ratings data after dropping both unpopular movies and inactive users: %s' %
                      str(active_users.shape))

        # pivot ratings into movie features
        self.movie_features = self.df_ratings.pivot(
            index='movieId',
            columns='userId',
            values='rating'
        ).fillna(0)
        self.movie_features_sparse = csr_matrix(self.movie_features.values)

        # make an object for the NearestNeighbors Class.
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        # fit the dataset
        self.model_knn.fit(self.movie_features_sparse)

    def fuzzy_matching(self, hashmap, fav_movie):
        """
        return the closest match via fuzzy ratio.
        If no match found, return None
        Parameters
        ----------
        hashmap: dict, map movie title name to index of the movie in data
        fav_movie: str, name of user input movie
        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]

    def make_recommendation(self, fav_movie, n_recommendations=10):
        mapper = {
            movie: i for i, movie in
            enumerate(list(self.df_movies.set_index('movieId').loc[self.movie_features.index].title))
        }
        # get input movie index
        print('You have input movie:', fav_movie)
        idx = self.fuzzy_matching(mapper, fav_movie)

        print('Recommendation system start to make inference')
        print('......\n')
        distances, indices = self.model_knn.kneighbors(self.movie_features_sparse[idx], n_neighbors=n_recommendations + 1)

        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        # get reverse mapper
        reverse_mapper = {v: k for k, v in mapper.items()}
        # print recommendations
        print('Recommendations for {}:'.format(fav_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], dist))


if __name__ == '__main__':
    my_movie = "Pulp Fiction"

    knn = KNearestNeighbours()
    knn.make_recommendation(my_movie)
