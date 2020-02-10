from data_handler import DataHandler


def lfm_data_splitter(ds):
    dh = DataHandler(ds=ds)
    train, test = dh.split_dataset_by_ratio([0.8, 0.2])
    return train, test
