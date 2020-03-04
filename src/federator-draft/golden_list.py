from definitions import ROOT_DIR
import logging.config
from knn_user import KNNUser
from lightfm_alg import LightFMAlg
from surprise_svd import SurpriseSVD

"""
Creates a "golden list" that will be the standard for future lists. Runs on both my custom KNN user-input alg and
LightFM's warp alg
"""


class GoldenList:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def generate_lists(self, user_id, data_path="/datasets/ml-latest-small", ds_ratings=None, min_rating=4.0,
                       num_of_recs=20, p_thresh=5, u_thresh=5, verbose=False):

        self.log.info("KNN Golden List:")
        golden_knn = KNNUser(user_id, data_path=data_path, p_thresh=p_thresh, u_thresh=u_thresh, ds_ratings=ds_ratings,
                             min_rating=min_rating).make_recommendation(num_of_recs, verbose=verbose)

        lfm_metric = "warp"  # warp or bpr
        self.log.info("LFM (%s) Golden List:" % lfm_metric)
        lfm_model = LightFMAlg(lfm_metric)
        golden_lfm = lfm_model.generate_rec(lfm_model.model, user_id, num_rec=num_of_recs)

        self.log.info("SVD Golden List:")
        svd = SurpriseSVD()
        golden_svd = svd.get_top_n(user_id, n=num_of_recs)

        return golden_knn, golden_lfm, golden_svd


if __name__ == '__main__':
    user_id = 1
    golden_lists = GoldenList().generate_lists(user_id, num_of_recs=-1)
