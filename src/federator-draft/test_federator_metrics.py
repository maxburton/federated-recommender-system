from lightfm_alg import LightFMAlg
from surprise_svd import SurpriseSVD
from data_handler import DataHandler
from definitions import ROOT_DIR
import helpers

import matplotlib.pyplot as plt


def test_alg_times():
    dh = DataHandler(filename=ROOT_DIR + "/datasets/ml-25m/ratings.csv")
    dh.dataset = dh.sort_dataset_randomly()
    # Test benchmark times
    ratings_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    for i in ratings_sizes:
        ds = dh.dataset[:i]
        user = ds[:, 0][0]
        #lfm = LightFMAlg(ds=dh.dataset, labels_ds="/datasets/ml-latest-small/movies.csv")
        #lfm.generate_rec(user)
        filename = "test_metrics_%d" % i
        svd = SurpriseSVD(ds=ds, sl_filename=filename,
                          movies_filename="/datasets/ml-25m/movies.csv")
        svd.get_top_n(user)


def plot_thresholds(rating_threshold, ratings, users, items):
    fig = plt.figure()
    fig.set_size_inches(6.4, 2.4)
    ax = plt.subplot(111)
    plt.title("Distribution of Number of Ratings, Users and Movies by Threshold")
    plt.ylabel("Number Above Threshold")
    plt.xlabel("Rating Threshold")

    #ax.plot(rating_threshold, ratings, label="total ratings")
    ax.plot(rating_threshold, users, label="users")
    ax.plot(rating_threshold, items, label="movies")
    ax.legend()
    # Put a legend below current axis
    save_filename = "zipfs_law.pdf"
    fig.savefig(save_filename, format="pdf", bbox_inches='tight')
    fig.show()


def calculate_thresholds():
    dh = DataHandler(filename=ROOT_DIR + "/datasets/ml-25m/ratings.csv")
    thresholds = [0, 10, 25, 50, 100, 250, 500, 750, 1000]
    ratings = []
    users = []
    items = []
    for t in thresholds:
        _, _, results = helpers.remove_below_threshold_user_and_items(dh.dataset, u_thresh=t, i_thresh=t)
        ratings.append(results[1])
        users.append(results[3])
        items.append(results[5])
    plot_thresholds(thresholds, ratings, users, items)

"""
Run methods
"""
#test_alg_times()
calculate_thresholds()
