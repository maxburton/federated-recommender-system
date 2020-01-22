from fuzzywuzzy import fuzz
import logging.config
from definitions import ROOT_DIR


def fuzzy_matching(hashmap, fav_movie, verbose=False):
    """
    return the closest match via fuzzy ratio.
    If no match found, return None
    Parameters
    ----------
    hashmap: dict, map movie title name to index of the movie in data
    fav_movie: str, name of user input movie
    Return
    ------
    index of the closest match
    """
    logging.config.fileConfig(ROOT_DIR + "/logging.conf", disable_existing_loggers=False)
    log = logging.getLogger(__name__)
    match_tuple = []
    # get match
    for title, idx in hashmap.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        log.info('Oops! No match is found')
    else:
        if verbose:
            log.info('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
        return match_tuple[0][1]
