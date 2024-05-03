import os.path
import fnmatch
from os import walk
import re
from typing import List

import numpy as np

# ---------------------------------------------------------------------------- #
#                         Useful code to open the data                         #
# ---------------------------------------------------------------------------- #

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

# ---------------------------------------------------------------------------- #
#                             Geometry-related code                            #
# ---------------------------------------------------------------------------- #

# TPC center coordinates
TPC_X = 0.
TPC_Y = -150.
TPC_Z = 1486.
# TPC measurements
TPCRadius = 277.02
TPCLength = 259.00
# TPC fiducial volume
TPCFidRadius = TPCRadius - 50.0
TPCFidLength = TPCLength - 50.0

# Define central points of the TPC left and right ends
pt1 = np.array([TPC_X-TPCLength, TPC_Y, TPC_Z])
pt2 = np.array([TPC_X+TPCLength, TPC_Y, TPC_Z])
# Same for fiducial volume
pt1_fid = np.array([TPC_X-TPCFidLength, TPC_Y, TPC_Z])
pt2_fid = np.array([TPC_X+TPCFidLength, TPC_Y, TPC_Z])

# Define a function to check if point is in TPC volume
def points_in_cylinder(pt1: np.array, pt2: np.array, r: float, q: np.array) -> bool:
    """Checks whether or not a point is contained inside a given cylinder

    Args:
        pt1 (np.array): central point of the left end of the cylinder
        pt2 (np.array): central point of the right end of the cylinder
        r (float):      radius of the cylinder
        q (np.array):   point to check

    Returns:
        bool: True if point is in cylinder, otherwise False
    """

    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    n = q.shape[0]
    return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const