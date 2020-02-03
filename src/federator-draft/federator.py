from definitions import ROOT_DIR
import logging.config
from golden_list import GoldenList
from individual_splits import IndividualSplits


class Federator:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    def __init__(self):
        user_id = 1
        golden_knn, golden_lfm = GoldenList().generate_lists(user_id)
        split_scores_knn, split_scores_lfm = IndividualSplits().run_on_splts(golden_knn, golden_lfm)
