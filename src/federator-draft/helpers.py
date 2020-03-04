import itertools
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm.data import Dataset
import matplotlib.pyplot as plt
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


# TODO: assign non-binary scores (i.e. higher the relevancy = higher relevancy value)
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


def order_top_k_items(golden, predicted, k=10, filename="/datasets/ml-latest-small/movies.csv"):
    title2id = generate_movietitle2id_mapper(filename=filename)
    golden_ids = {}
    for i in range(len(golden)):
        golden_ids[title2id[golden[i][1]]] = golden[i]
    predicted = predicted[:k]
    relevance_values = []
    for title in predicted:
        golden_score = golden_ids[title2id[title[1]]][2]  # aka relevance score
        relevance_values.append(golden_score)
    return np.array(relevance_values), predicted


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
        dtype={'movieId': 'int32', 'title': 'str'})

    titles = df_movies.title
    ids = df_movies.movieId
    return {k: v for k, v in zip(titles, ids)}


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


"""
NDCG Metrics
(from https://gist.github.com/bwhite/3726239)
"""

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max