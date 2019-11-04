class Error(Exception):
    """Base class for other custom exceptions"""
    pass


class MissingArgumentException(Error):
    """Raised when a required argument is not provided"""
    pass
