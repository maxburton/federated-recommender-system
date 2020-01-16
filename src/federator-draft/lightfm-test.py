from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import logging.config
from lightfm.datasets import fetch_movielens
import data_handler
from definitions import ROOT_DIR


class LightFMAlg:
    logging.config.fileConfig(ROOT_DIR + 'logging.conf', disable_existing_loggers=False)
    log = logging.getLogger(__name__)
    dataset = [1, 2, 3]
    data_slicer = data_handler.DataSlicer(dataset)

    movielens = fetch_movielens()

    train = movielens['train']
    test = movielens['test']

    # WARP Loss Function
    model = LightFM(learning_rate=0.05, loss='warp')

    model.fit_partial(train, epochs=10)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10, train_interactions=train).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test, train_interactions=train).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    # Suggests that WARP is superior to BPR

