import os.path
import fnmatch
from os import walk
import re
from typing import List

def get_datafile_list(data_path: str, match_exprs: List[str] = ["*.root"]) -> List[str]:
    """ Get a list with the names of the files in a certain directory
        containing a substring from a list.

    Args:
        data_path (str): Path to directory of interest.
        match_exprs (list, optional): List of expressions to test. Defaults to ["*.root"].

    Returns:
        list: List of file names in directory with matching substring(s).
    """

    files = []
    for m in match_exprs:
        files += fnmatch.filter(next(walk(data_path), (None, None, []))[2], m)  # [] if no file

    return sorted(files, reverse=True, key=lambda f: os.path.getmtime(os.path.join(data_path, f)))

def sorted_nicely(files: List[str]) -> List[str]:
    """ Sort the given iterable in the way that humans expect.

    Args:
        files (list): List of strings to sort (typically list of file names).

    Returns:
        list: Sorted list.
    """

    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(files, key = alphanum_key)