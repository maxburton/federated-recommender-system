import numpy as np
from lightfm import LightFM
from definitions import ROOT_DIR
import logging.config
from data_handler import DataHandler
import random
import helpers


class LightFMAlg:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self, loss_type, ds=None, labels=None, labels_ds=None,
                 normalisation=None, learning_rate=0.05, min_rating=4.0):
        if ds is None:
            ds_path = ROOT_DIR + "/datasets/ml-latest-small/ratings.csv"
            dh = DataHandler(filename=ds_path)
        elif type(ds) is str:
            ds_path = ROOT_DIR + ds
            dh = DataHandler(filename=ds_path)
        else:
            dh = DataHandler(ds=ds)

        # Normalise ratings
        if normalisation:
            normalised_ds = dh.get_dataset()
            normalised_ds[:, 2] = normalisation(normalised_ds[:, [0, 2]])
            dh.set_dataset(normalised_ds)

            # convert min_rating to fit with normalised scores
            lower = np.min(normalised_ds[:, 2])
            upper = np.max(normalised_ds[:, 2])
            min_rating = min_rating / (5 / upper - lower)

        self.labels = labels
        if labels is None:
            if labels_ds is None:
                ds_path = ROOT_DIR + "/datasets/ml-latest-small/movies.csv"
            else:
                ds_path = ROOT_DIR + labels_ds
            self.labels = DataHandler(filename=ds_path).get_dataset()[:, 0:2]

        train_raw, test_raw = dh.split_dataset_by_ratio([0.8, 0.2])

        num_users, num_items, users, items = helpers.get_dimensions(helpers.parse(train_raw),
                                                                    helpers.parse(test_raw))

        self.item_inv_mapper = helpers.generate_mapper(items)
        self.user_inv_map = helpers.generate_mapper(users)

        # remove labels that aren't present in dataset (-1 to zero index)
        self.labels = self.labels[:, 1][np.isin(self.labels[:, 0]-1, items)]

        self.train = helpers.build_interaction_matrix(num_users, num_items, helpers.parse(train_raw), min_rating,
                                                      self.item_inv_mapper, self.user_inv_map)

        self.log.info("Generating LFM model...")
        self.model = LightFM(learning_rate=learning_rate, loss=loss_type)
        self.model.fit(self.train, epochs=30, num_threads=2)

    #  Prints the top n user rated items
    @staticmethod
    def print_known(user_id, known, num_known=5):
        print("User %d likes:" % (user_id+1))
        for i in range(num_known):
            if len(known) > i:
                print("%d: %s" % (i+1, known[i]))
            else:
                print("User has no more favourites!")
                break

    # Generates random recs if user does not exist
    def generate_random_recs(self, n=10, verbose=False):
        n_users, n_items = self.train.shape
        scores = self.model.predict(0, np.arange(n_items))
        top_items = self.labels[np.argsort(-scores)]
        scores = scores[np.argsort(-scores)]
        raw_recs = np.random.choice(top_items, size=n)
        min_score = np.min(scores)
        max_score = np.max(scores)
        recs = []
        for i in range(raw_recs.shape[0]):
            recs.append([i + 1, raw_recs[i], random.uniform(min_score, max_score)])
        if verbose:
            self.log.info(repr(self.train))
            helpers.pretty_print_results(self.log, recs, user_id+1)
        return np.array(recs)

    #  Generates recs for LFM.
    def generate_rec(self, model, raw_user_id, num_known=5, num_rec=10, verbose=False):
        raw_user_id -= 1  # lfm is zero indexed
        try:
            user_id = self.user_inv_map[raw_user_id]
        except KeyError:
            self.log.info("User doesn't exist in dataset, returning random recs")
            return self.generate_random_recs(n=num_rec, verbose=verbose)
        n_users, n_items = self.train.shape
        # for user_id in user_ids:  # if i want to support multi user entry in the future
        known_positives = self.labels[self.train.tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = self.labels[np.argsort(-scores)]
        scores = scores[np.argsort(-scores)]

        recs = []
        num_rec_internal = num_rec
        if num_rec_internal == -1:
            num_rec_internal = len(top_items)

        # Since LFM sometimes recommends items the user has already rated, we strip this out so they are not recommended
        current_i = 0
        valid_recs = 0
        while valid_recs < num_rec_internal:
            # If num_recs = -1 or if it is greater than the number of valid recs, we simply return as many as there are
            try:
                current_item = top_items[current_i]
                if current_item not in known_positives:
                    recs.append([valid_recs + 1, current_item, scores[current_i]])
                    valid_recs += 1
                current_i += 1
            except IndexError:
                if num_rec == -1:
                    self.log.info("All valid items retrieved!")
                    break
                else:
                    self.log.warning("Number of recommendations out of bounds, only returning %d items. "
                                     "Try lowering the number of recommendations desired in the future." % valid_recs)
                    break

        if verbose:
            self.log.info(repr(self.train))
            self.print_known(user_id, known_positives, num_known=num_known)
            helpers.pretty_print_results(self.log, recs, user_id+1)
        return np.array(recs)


if __name__ == "__main__":
    user_id = 1
    alg_bpr = LightFMAlg("bpr")  # warp or bpr
    alg_bpr.generate_rec(alg_bpr.model, user_id, num_rec=20)

    ds_path = ROOT_DIR + "/datasets/ml-25m/ratings.csv"
    dh = DataHandler(filename=ds_path)
    dh = helpers.remove_below_threshold_user_and_items(dh, u_thresh=500, i_thresh=500)
    alg_warp = LightFMAlg("warp", ds=dh.get_dataset(), labels_ds="/datasets/ml-25m/movies.csv")  # warp or bpr
    alg_warp.generate_rec(alg_warp.model, user_id, num_rec=20)
