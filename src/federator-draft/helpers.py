import itertools
from sklearn.metrics import dcg_score
import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from definitions import ROOT_DIR


def pretty_print_results(log, results, user_id):
    log.info('Recommendations for user {}:'.format(user_id))
    for row in results:
        if len(row) == 4:
            log.info('{0}: {1}, with score {2} from the {3} algorithm'.format(row[0], row[1], row[2], row[3]))
        else:
            log.info('{0}: {1}, with score {2}'.format(row[0], row[1], row[2]))


def convert_np_to_pandas(pd, a, first_col=0, last_col=3):
    return pd.DataFrame(data=a[:, first_col:last_col], columns=['userId', 'movieId', 'rating'])


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


# DEPRECATED HELPERS:

def get_relevant_values(recs, golden):
    relevant_values = []
    golden_movie_names = golden[:, 1]
    from_name_to_rank = {k: v for v, k in enumerate(golden_movie_names, start=1)}
    for rec in recs:
        r_value = 0
        rank, movie_name, _ = rec
        if movie_name in golden_movie_names:
            distance = abs(from_name_to_rank[movie_name] - int(rank))
            r_value = 1 + 2/((len(golden_movie_names)/2) + distance/len(golden_movie_names)/2)
        relevant_values.append(r_value)
    return relevant_values


def get_relevant_values_2(golden, predicted, k=10):
    golden_arr = np.arange(1, k+1)
    predicted_arr = np.full(k, k*3)
    predicted_movie_names = predicted[:, 1]
    for i in range(k):
        title = golden[i][1]
        predicted_rank = predicted[i][0]
        if title in predicted_movie_names:
            predicted_arr[i] = predicted_rank
    return np.expand_dims(golden_arr, axis=0), np.expand_dims(predicted_arr, axis=0)
