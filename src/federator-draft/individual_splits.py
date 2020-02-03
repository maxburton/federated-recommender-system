from definitions import ROOT_DIR
import logging.config
from knn_user import KNNUser
from lightfm_alg import LightFMAlg

"""
Runs each alg on individual splits, then calculates their score against their respective golden list
"""


class IndividualSplits:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    #TODO: Create the splits, run each alg on the splits, and for each split calculate score against golden list
    def run_on_splts(self):
        return
