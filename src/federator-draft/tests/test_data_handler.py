import unittest
import numpy as np
import logging
import logging.config
from definitions import ROOT_DIR
from exceptions import MissingArgumentException, InvalidShapeException

from data_handler import DataHandler


class TestDataHandler(unittest.TestCase):
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    ML100K_PATH = ROOT_DIR + "/datasets/ml-100k/u.data"

    def setUp(self):
        print("Running test: %s" % self._testMethodName)

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

    """
    Tests the init method
    """

    def test_sort_dataset_by_col_raises_missing_argument_exception(self):
        self.assertRaises(MissingArgumentException, DataHandler)

    def test_init_reads_from_file_with_correct_matrix(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        self.assertTrue(np.array_equal([196, 242, 3, 881250949], data.get_dataset()[0]))
        self.assertTrue(np.array_equal([253, 465, 5, 891628467], data.get_dataset()[7]))
        self.assertTrue(np.array_equal([12, 203, 3, 879959583], data.get_dataset()[-1]))

        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=400000)
        self.assertEqual(data.get_dataset()[0][399999], 879959583)

    def test_invalid_cols_raises_exceptions(self):
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=7)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=3)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=99)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=400001)
        self.assertRaises(ZeroDivisionError, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=0)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=-4)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=-7)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=-400000)
        self.assertRaises(InvalidShapeException, DataHandler, filename=self.ML100K_PATH, dtype=np.uint32, cols=-800001)

    """
    Tests the extract whole entries method
    """

    def test_movie_ids_are_returned(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        extracted = data.extract_whole_entries(5, col=1)
        self.assertTrue((extracted == [293, 5, 3, 888906576]).all(axis=1).any())  # tests if row in array
        self.assertTrue((extracted == [99, 4, 5, 886519097]).all(axis=1).any())
        self.assertTrue((extracted == [287, 1, 5, 875334088]).all(axis=1).any())
        self.assertFalse((extracted == [166, 346, 1, 886397596]).all(axis=1).any())
        self.assertFalse((extracted == [200, 222, 5, 876042340]).all(axis=1).any())
        self.assertFalse((extracted == [71, 6, 3, 880864124]).all(axis=1).any())

        id_array = extracted[:, 1]
        self.assertTrue(np.isin(id_array, [1, 2, 3, 4, 5]).all())
        self.assertFalse(np.isin(id_array, [2, 3, 4, 5]).all())
        self.assertFalse(np.isin(id_array, [1, 2, 3, 4]).all())

    def test_movie_ids_are_deleted_from_dataset(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)

        self.assertTrue((data.get_dataset() == [293, 5, 3, 888906576]).all(axis=1).any())  # tests if row in ds
        data.extract_whole_entries(5, col=1, delete=False)
        self.assertTrue((data.get_dataset() == [293, 5, 3, 888906576]).all(axis=1).any())  # tests if row still in ds
        data.extract_whole_entries(5, col=1)
        self.assertFalse((data.get_dataset() == [293, 5, 3, 888906576]).all(axis=1).any())  # tests if row now not in ds

    """
    Tests the split methods
    """

    def test_splits_are_correct_length(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        splits = 10
        split = data.split_dataset(splits)  # nice, even split
        self.assertTrue(len(split) == splits)
        self.assertTrue(len(split[0]) == 10000)
        self.assertTrue(len(split[5]) == 10000)
        self.assertTrue(len(split[-1]) == 10000)

        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        splits = 7
        split = data.split_dataset(splits)  # ugly split
        self.assertTrue(len(split) == splits)
        first_split_len = len(split[0])
        for i in range(splits):  # test that each split is +- 1 of the original split
            self.assertTrue(first_split_len-1 <= len(split[i]) <= first_split_len+1)

    def test_split_intermittently_splits_correctly(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        splits = 5
        split = data.split_dataset_intermittently(splits)
        self.assertTrue((split[0][1] == [298, 474, 4, 884182806]).all())
        self.assertTrue((split[-1][-1] == [12, 203, 3, 879959583]).all())

    def test_sort_then_split(self):
        data = DataHandler(filename=self.ML100K_PATH, dtype=np.uint32, cols=4)
        sort = data.sort_dataset_by_col(1)
        splits = 10
        split = data.split_dataset_intermittently(splits, ds=sort)
        self.assertTrue(np.count_nonzero(sort[:, 1] == 1) >= (splits * 2))
        for i in range(splits):
            self.assertTrue(np.count_nonzero(split[i][:, 1] == 1) > 1)


if __name__ == '__main__':
    unittest.main()
