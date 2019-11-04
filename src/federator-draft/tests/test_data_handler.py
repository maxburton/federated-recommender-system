import unittest
import numpy as np
import logging
import logging.config
from definitions import ROOT_DIR
from exceptions import MissingArgumentException

from data_handler import DataHandler


class TestDataHandler(unittest.TestCase):
    logging.config.fileConfig(ROOT_DIR + "/logging.conf")
    log = logging.getLogger(__name__)

    """
    Tests the sort_dataset_by_col method
    """
    def test_sort_dataset_by_col_correctly_sorts1(self):
        a = np.arange(12)
        a = np.append(a, np.zeros(4)).reshape((4, 4))
        data_handler = DataHandler(ds=a)
        data = data_handler.sort_dataset_by_col(1)
        # logging.info(data)

        expected_data = np.zeros(4)
        expected_data = np.append(expected_data, np.arange(12)).reshape((4, 4))
        # logging.info(expected_data)

        self.assertTrue(np.array_equal(data, expected_data))

    def test_sort_dataset_by_col_correctly_sorts2(self):
        b = np.arange(256).reshape((256, 1))
        np.random.shuffle(b)
        data_handler = DataHandler(ds=b)
        data = data_handler.sort_dataset_by_col(0)
        # logging.info(data)

        expected_data = np.arange(256).reshape((256, 1))
        # logging.info(expected_data)

        self.assertTrue(np.array_equal(data, expected_data))

    def test_sort_dataset_by_col_raises_missing_argument_exception(self):
        self.assertRaises(MissingArgumentException, DataHandler)


if __name__ == '__main__':
    unittest.main()
