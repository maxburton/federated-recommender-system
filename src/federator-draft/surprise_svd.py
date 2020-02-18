from surprise import Dataset, Reader, SVD
from surprise.model_selection.validation import cross_validate
from definitions import ROOT_DIR
import logging.config
from collections import defaultdict


class SurpriseSVD:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, ds=None):
        reader = Reader(line_format='user item rating timestamp', sep=",", skip_lines=1)
        if ds is None:
            data = Dataset.load_from_file(ROOT_DIR+"/datasets/ml-latest-small/ratings.csv", reader=reader)
        else:
            reader = Reader(line_format='user item rating timestamp', sep=",")
            data = Dataset.load_from_df(ds, reader=reader)
        algo = SVD()
        results = cross_validate(algo, data, measures=['RMSE', 'MAE'])
        print(repr(results))

        trainset = data.build_full_trainset()
        testset = trainset.build_anti_testset()

        algo.fit(trainset)
        self.predictions = algo.test(testset)

    def get_top_n(self, user_id, n=10, verbose=True):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
            verbose(bool): if True, prints the top n results

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # TODO: Change so only 1 user's results are returned
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        if verbose:
            for uid, user_ratings in top_n.items():
                print(uid, [iid for (iid, _) in user_ratings])

        return top_n


if __name__ == '__main__':
    user_id = 1
    svd = SurpriseSVD()
    svd.get_top_n(user_id, n=10)
