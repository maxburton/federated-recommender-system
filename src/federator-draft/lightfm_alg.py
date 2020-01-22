import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from definitions import ROOT_DIR
import logging.config
from data_handler import DataHandler
import helpers


class LightFMAlg:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    model = None
    data = None

    def __init__(self, loss_type, learning_rate=0.05, min_rating=4.0):
        data = fetch_movielens(min_rating=min_rating)   # Can remove min rating
        #self.log.info(data)
        self.log.info(repr(data['train']))
        self.log.info(repr(data['test']))

        model = LightFM(learning_rate=learning_rate, loss=loss_type)
        model.fit(data["train"], epochs=30, num_threads=2)

        self.model = model
        self.data = data

    @staticmethod
    def print_known(user_id, known, num_known=5):
        print("User %s likes:"  % user_id)
        for i in range(num_known):
            print("%d: %s" % (i+1, known[i]))

    def generate_rec(self, model, data, user_id, num_known=5, num_rec=10):
        n_users, n_items = data["train"].shape
        #for user_id in user_ids:  # if i want to support multi user entry in the future
        known_positives = data["item_labels"][data["train"].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data["item_labels"][np.argsort(-scores)]
        scores = scores[np.argsort(-scores)]

        recs = []
        for i in range(num_rec):
            recs.append([i+1, top_items[i], scores[i]])
        self.print_known(user_id, known_positives, num_known=num_known)
        helpers.pretty_print_results(self.log, recs, user_id)
        return recs


if __name__ == "__main__":
    alg_bpr = LightFMAlg("bpr")  # warp or bpr
    alg_bpr.generate_rec(alg_bpr.model, alg_bpr.data, 1, num_rec=20)

    alg_warp = LightFMAlg("warp")  # warp or bpr
    alg_warp.generate_rec(alg_warp.model, alg_warp.data, 1, num_rec=20)
