class Error(Exception):
    """Base class for other custom exceptions"""
    pass


class MissingArgumentException(Error):
    """Raised when a required argument is not provided"""
    pass


class InvalidShapeException(Error):
    """Raised when an invalid shape is provided"""
    pass


class InvalidRatioSumException(Error):
    """Raised when the sum of elements in ratios is not equal to 1.0"""
    pass


class InvalidRatioIndicesException(Error):
    """Raised when the number of indices is not equal to the number of partitions"""
    pass
