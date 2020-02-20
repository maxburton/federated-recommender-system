import numpy as np
from lightfm import LightFM
from definitions import ROOT_DIR
import logging.config
from data_handler import DataHandler
import helpers


class LightFMAlg:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, loss_type, ds=None, labels=None, learning_rate=0.05, min_rating=4.0):
        if ds is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
            dh = DataHandler(filename=ds_path)
        else:
            dh = DataHandler(ds=ds)

        self.labels = labels
        if labels is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/movies.csv"
            self.labels = DataHandler(filename=ds_path).get_dataset()[:, 0:2]

        train_raw, test_raw = dh.split_dataset_by_ratio([0.8, 0.2])

        num_users, num_items, users, items = helpers.get_dimensions(helpers.parse(train_raw),
                                                                    helpers.parse(test_raw))

        self.item_inv_mapper = helpers.generate_mapper(items)
        self.user_inv_map = helpers.generate_mapper(users)
        self.labels = self.labels[:, 1][np.isin(self.labels[:, 0]-1, items)]  # remove labels that aren't present in dataset (-1 to zero index)

        self.train = helpers.build_interaction_matrix(num_users, num_items, helpers.parse(train_raw), min_rating,
                                                      self.item_inv_mapper, self.user_inv_map)
        self.log.info(repr(self.train))

        self.model = LightFM(learning_rate=learning_rate, loss=loss_type)
        self.model.fit(self.train, epochs=30, num_threads=2)

    @staticmethod
    def print_known(user_id, known, num_known=5):
        print("User %d likes:" % (user_id+1))
        for i in range(num_known):
            if len(known) > i:
                print("%d: %s" % (i+1, known[i]))
            else:
                print("User has no more favourites!")
                break

    def generate_rec(self, model, user_id, num_known=5, num_rec=10):
        n_users, n_items = self.train.shape
        # for user_id in user_ids:  # if i want to support multi user entry in the future
        known_positives = self.labels[self.train.tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = self.labels[np.argsort(-scores)]
        scores = scores[np.argsort(-scores)]

        recs = []
        if num_rec == -1:
            num_rec = len(top_items)

        for i in range(num_rec):
            recs.append([i+1, top_items[i], scores[i]])
        self.print_known(user_id, known_positives, num_known=num_known)
        helpers.pretty_print_results(self.log, recs, user_id+1)
        return np.array(recs)


if __name__ == "__main__":
    alg_bpr = LightFMAlg("bpr")  # warp or bpr
    alg_bpr.generate_rec(alg_bpr.model, 0, num_rec=20)

    alg_warp = LightFMAlg("warp")  # warp or bpr
    alg_warp.generate_rec(alg_warp.model, 0, num_rec=20)
