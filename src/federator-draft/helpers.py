import itertools
from sklearn.metrics import dcg_score
import math
import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from collections import defaultdict

from definitions import ROOT_DIR


def pretty_print_results(log, results, user_id):
    log.info('Recommendations for user {}:'.format(user_id))
    for row in results:
        if len(row) == 4:
            log.info('{0}: {1}, with score {2} from the {3} algorithm'.format(row[0], row[1], row[2], row[3]))
        else:
            log.info('{0}: {1}, with score {2}'.format(row[0], row[1], row[2]))


def convert_np_to_pandas(a, columns=None):
    if columns is None:
        columns = ['userId', 'movieId', 'rating']
    first_col = 0
    last_col = len(columns)
    return pd.DataFrame(data=a[:, first_col:last_col], columns=columns)


def remove_below_threshold_user_and_items(dh, u_thresh=0, i_thresh=0):
    df = convert_np_to_pandas(dh.get_dataset(), columns=['userId', 'movieId', 'rating', 'timestamp'])
    num_users = df['userId'].value_counts().size
    num_items = df['movieId'].value_counts().size
    print("Shape before filter: {0}, no. users: {1}, no. items: {2}".format(str(df.shape), num_users, num_items))
    df = df.groupby('userId').filter(lambda x: len(x) > u_thresh)
    df = df.groupby('movieId').filter(lambda x: len(x) > i_thresh)
    surviving_users = df['userId'].value_counts()
    surviving_items = df['movieId'].value_counts()
    print("Shape after filter: {0}, no. users: {1}, no. items: {2}".format(str(df.shape), surviving_users.size,
                                                                           surviving_items.size))
    dh.set_dataset(df.to_numpy())
    return dh, surviving_users


def get_users_with_min_ratings(users, min_ratings=10):
    return users[users > min_ratings]


""" 
Returns the respective golden scores for the top k predicted items

in_order will provide scores that sklearn's dcg method will take to rank the relevance values in the order provided
"""
def order_top_k_items(golden, predicted, log, k=10, in_order=False, filename="/datasets/ml-latest-small/movies.csv"):
    title2id = generate_movietitle2id_mapper(filename=filename)
    golden_ids = {}
    len_golden = len(golden)

    # Get all movie ids in the golden list, mapped to their golden ranking/score
    for i in range(len_golden):
        golden_ids[title2id[golden[i][1]]] = golden[i]
    predicted = predicted[:k]

    # for each predicted item, get its respective score from the golden list
    relevance_values = []
    for prediction in predicted:
        try:
            # golden_score = golden_ids[title2id[title[1]]][2]  # golden list score
            # We use the position in the golden list's ranking as a relevance value, since scores are mostly similar
            #golden_rank = 1 - float(golden_ids[title2id[title[1]]][0]) / len_golden
            rank = float(golden_ids[title2id[prediction[1]]][0])
            relevance_values.append(rank_scorer(rank, k=k))
        except KeyError:
            log.warning("This item doesn't exist in the golden list, assigning score of 0")
            relevance_values.append(0)
            continue

    # (We add another dimension to play nice with sklearn's dcg method)
    if in_order:
        return np.array([relevance_values]).astype(float), np.array([range(k+1, 1, -1)])
    else:
        return np.array([relevance_values]).astype(float), np.array([predicted[:, 2]]).astype(float)


def best_dcg_score(k=10):
    relevance_values = []
    for i in range(k):
        relevance_values.append(rank_scorer(i, k=k))
    scores = np.arange(1, k+1)
    return dcg_score(np.array([relevance_values]).astype(float), np.array([scores]).astype(float), k=k)


def rank_scorer(rank, k=10, var=None):
    if rank < k // 5:
        golden_rank = 2
    elif rank < k:
        golden_rank = 1
    else:
        golden_rank = 0

    return golden_rank


def pick_random(recs, n):
    rand_indices = np.random.choice(recs.shape[0], n, replace=False)
    return recs[rand_indices]


def scale_scores(scores):
    min_max_scaler = MinMaxScaler()
    scaled_score = min_max_scaler.fit_transform(scores.reshape(-1, 1).astype(float))
    return scaled_score


def lfm_data_mapper(ds):
    dataset = Dataset()
    users = ds["userId"].to_numpy()
    movies = ds["movieId"].to_numpy()
    dataset.fit(users, movies)
    (interactions, weights) = dataset.build_interactions(zip(users, movies))
    print(repr(interactions))
    return interactions


def parse(data):

    for line in data:

        uid, iid, rating, timestamp = [int(x) for x in line]

        # Subtract one from ids to shift
        # to zero-based indexing
        yield uid - 1, iid - 1, rating, timestamp


def get_dimensions(train_data, test_data):

    uids = set()
    iids = set()

    for uid, iid, _, _ in itertools.chain(train_data,
                                          test_data):
        uids.add(uid)
        iids.add(iid)

    rows = len(uids)
    cols = len(iids)

    return rows, cols, np.array(sorted(list(uids))), np.array(sorted(list(iids)))


def build_interaction_matrix(rows, cols, data, min_rating, item_mapper, user_mapper):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            mat[user_mapper[uid], item_mapper[iid]] = rating  # mappers required as some users/items might not exist

    return mat.tocoo()


def generate_mapper(lst):
    mapper = {k: v for k, v in enumerate(lst)}
    inverted_mapper = {v: k for k, v in mapper.items()}

    return inverted_mapper


def generate_movietitle2id_mapper(filename="/datasets/ml-latest-small/movies.csv"):
    filepath = ROOT_DIR + filename
    df_movies = pd.read_csv(
        filepath,
        usecols=['movieId', 'title'],
        dtype={'movieId': 'int', 'title': 'str'})

    return {k: v for k, v in zip(df_movies["title"], df_movies["movieId"])}


# We can't simply reverse the previous mapper because duplicate titles exist in the dataset (movies.csv)
def generate_id2movietitle_mapper(filename="/datasets/ml-latest-small/movies.csv"):
    filepath = ROOT_DIR + filename
    df_movies = pd.read_csv(
        filepath,
        usecols=['movieId', 'title'],
        dtype={'movieId': 'int', 'title': 'str'})

    return {k: v for k, v in zip(df_movies["movieId"], df_movies["title"])}


def create_scatter_graph(title, x_label, y_label, key_labels, colors, *args, ymin=0, ymax=1.2, x=None, s=None, alpha=None):
    if x is None:
        x = np.linspace(0, len(args[0]), len(args[0]))
        for i in range(len(args)):
            plt.scatter(x, args[i], s=s, c=colors[i], label=key_labels[i], alpha=alpha)
    else:
        for i in range(len(args)):
            plt.scatter(x[i], args[i], s=s, c=colors[i], label=key_labels[i], alpha=alpha)
    plt.legend()
    x_max = 0
    for arg in args:
        x_max += len(arg)
    plt.axis([0, x_max+1, ymin, ymax])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


"""From https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py

Return precision and recall at k metrics for each user in the SVD alg.
"""
def svd_precision_recall_at_k(predictions, k=10, threshold=4.65):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rec_k / k

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


"""
Return a dictionary with every user id and their mean rating across all their ratings in a dataset
"""
def get_avg_user_ratings(ratings, variance=False):
    user_ids = ratings[:, 0]
    user_ratings = ratings[:, 1]
    avg_user_ratings = {}
    user_variances = {}
    unique_user_ids = np.unique(user_ids)
    for user_id in unique_user_ids:
        avg_user_rating = np.mean(user_ratings[user_ids == user_id])
        avg_user_ratings[user_id] = avg_user_rating
        if variance:
            user_variance = math.sqrt(np.sum((user_ratings[user_ids == user_id] - avg_user_rating)**2))
            if user_variance != 0:
                user_variances[user_id] = user_variance
            else:
                user_variances[user_id] = 1.0
    return user_ids, user_ratings, avg_user_ratings, user_variances


"""
Returns a normalised rating, penalised by the average ratings of each user. Can produce negative ratings.
Effective against users that rate consistently high or low
"""
def shifted_normalisation(ratings):
    # Get each user's average rating
    user_ids, user_ratings, avg_user_ratings, _ = get_avg_user_ratings(ratings)

    normalised_ratings = np.array([user_ratings[i] - avg_user_ratings[user_ids[i]] for i in range(ratings.shape[0])])
    scaled_normalised_ratings = MinMaxScaler(feature_range=(0, 5)).fit_transform(normalised_ratings.reshape(-1, 1)).flatten()
    return scaled_normalised_ratings


"""
Returns a normalised rating, fit to a gaussian curve. Can produce negative ratings.
Effective against users that rate very varied scores or very narrow scores
"""
def gaussian_normalisation(ratings):
    user_ids, user_ratings, avg_user_ratings, user_variance = get_avg_user_ratings(ratings, variance=True)

    normalised_ratings = np.array([(user_ratings[i] - avg_user_ratings[user_ids[i]]) / user_variance[user_ids[i]]
                                   for i in range(ratings.shape[0])])

    scaled_normalised_ratings = RobustScaler().fit_transform(normalised_ratings.reshape(-1, 1)).flatten()
    return scaled_normalised_ratings


"""
Returns a probability rating of whether or not the user will like an item based on their distribution of ratings

e.g. if a user often rates items '4', then it loses its effect and the user probably doesn't particularly like this
item. Equally, if the user has rated an item a '4' but there exists many items rated lower than or equal to
'4', the user will probably like it.

returns a float between 0 and 1.
"""
def decoupling_normalisation(ratings):
    user_ids = ratings[:, 0]
    user_ratings = ratings[:, 1]
    unique_user_ids = np.unique(user_ids)

    user_probabilities = {}
    user_lt_probabilities = {}

    for user_id in unique_user_ids:
        rating_probabilities = []
        rating_lt_probabilities = []
        users_ratings = user_ratings[user_ids == user_id]
        total_ratings = users_ratings.shape[0]

        for i in range(10):
            # Get all ratings that are equal to the current rating
            current_rating = 0.5 * (i+1)
            rating_count = users_ratings[users_ratings == current_rating].shape[0]

            # Find the probability distribution for each rating category
            rating_probabilities.append(rating_count / total_ratings)

            # Sum the current list of categories to find the probability distribution of being <= this rating category
            rating_lt_probabilities.append(sum(rating_probabilities))
        user_probabilities[user_id] = rating_probabilities
        user_lt_probabilities[user_id] = rating_lt_probabilities

    # convert rating into equivalent index
    normalised_ratings = np.array([user_lt_probabilities[user_ids[i]][int(user_ratings[i]*2 - 1)] -
                                   user_probabilities[user_ids[i]][int(user_ratings[i]*2 - 1)]/2
                                   for i in range(ratings.shape[0])])
    scaled_normalised_ratings = RobustScaler().fit_transform(normalised_ratings.reshape(-1, 1)).flatten()
    return scaled_normalised_ratings

