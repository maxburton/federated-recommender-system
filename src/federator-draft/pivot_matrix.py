import numpy as np
from scipy.sparse import csr_matrix
import logging.config
from definitions import ROOT_DIR

int32_max_size = 2147483647


class PivotMatrix:
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)

    movie_count = 0
    user_count = 0
    total_cells = 0

    def pivot(self, df):

        self.movie_count = df["movieId"].nunique()
        self.user_count = df["userId"].nunique()
        total_cells = self.movie_count * self.user_count
        self.log.info("movie count:%d user count:%d total cells:%d" % (self.movie_count, self.user_count, total_cells))

        if total_cells > int32_max_size // 4:
            return self.pivot_large(df)
        else:
            return self.pivot_small(df)

    def pivot_large(self, df):
        self.log.info("DF needs to be partitioned")
        partitions = ((self.total_cells // int32_max_size) + 1) + 8  # Additional partitions for performance
        partition_size = len(df) // partitions
        self.log.info("partitions:%d partition size:%d" % (partitions, partition_size))

        m_meta = MatrixMeta()

        for i in range(partitions):
            lower = i * partition_size
            upper = (i + 1) * partition_size
            if (i + 1) == partitions:
                upper = len(df)
            self.log.info("Partition row %d from lower:%d, upper:%d" % (i+1, lower, upper))
            movie_features = df.iloc[lower:upper].pivot(
                index='movieId',
                columns='userId',
                values='rating'
            ).fillna(0)

            # Convert to sparse matrix to save memory
            row_indices, col_indices = np.nonzero(movie_features.values)
            data = movie_features.values[row_indices, col_indices]
            m_meta.extend(data, row_indices, col_indices)

        # Build up the entire matrix from the partitions
        movie_features_sparse = csr_matrix(m_meta.get_meta(), shape=(self.movie_count, self.user_count))
        return movie_features_sparse

    def pivot_small(self, df):
        self.log.info("DF does not need to be partitioned")
        movie_features = df.pivot(
            index='movieId',
            columns='userId',
            values='rating'
        ).fillna(0)
        return csr_matrix(movie_features)


class MatrixMeta:
    all_data = []
    all_row_indices = []
    all_col_indices = []

    def extend(self, d, ri, ci):
        self.all_data.extend(d)
        self.all_row_indices.extend(ri)
        self.all_col_indices.extend(ci)

    def get_meta(self):
        return (self.all_data, (self.all_row_indices, self.all_col_indices))
