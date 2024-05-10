import os.path
import fnmatch
from os import walk
import re
from typing import List

import numpy as np
import pickle

from garanapy import datamanager
from garanapy import idle

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

def open_pickle_data(file: str) -> datamanager.DataManager:

    """ Open DataManager instance from pickle file

    Args:
        file (str): Path to pickle file with saved DataManager instance

    Returns:
        datamanager.DataManager: Previously stored instance of DataManager
    """

    loop = idle.Idle("Opening pickled data...", "Finished opening data!")
    loop.start_idle()
    with open(file, 'rb') as input:
            data_manager = pickle.load(input)
    loop.end_idle()
    return data_manager

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

# We're going to use this a lot, so better to have a shortcut
def in_fiducial(p: np.array) -> bool:
    return points_in_cylinder(pt1_fid, pt2_fid, TPCFidRadius, p)

# ---------------------------------------------------------------------------- #
#                              Useful definitions                              #
# ---------------------------------------------------------------------------- #

# Masses of the particles (in GeV)
m_electron     =   0.511*1e-3
m_muon         = 105.658*1e-3
m_neutral_pion = 134.977*1e-3
m_pion         = 139.570*1e-3
m_kaon         = 493.677*1e-3
m_proton       = 938.272*1e-3

particle_names = {11:   r"$e^{\pm}$",
                  13:   r"$\mu^{\pm}$",
                  22:   r"$\gamma$",
                  111:  r"$\pi^{0}$",
                  211:  r"$\pi^{\pm}$",
                  321:  r"$K^{\pm}$",
                  2212: r"$p$"}

particle_masses = {11:   m_electron,
                   13:   m_muon,
                   22:   0.0,
                   111:  m_neutral_pion,
                   211:  m_pion,
                   321:  m_kaon,
                   2212: m_proton}

def total_energy(pdg: int, momentum: float) -> float:

    """ Return total energy of particle from PDG number and momentum

    Args:
        pdg (int):        PDG code of particle
        momentum (float): momentum of particle (in GeV)

    Returns:
        float: total energy of particle (in GeV)
    """
    return np.sqrt(np.square(particle_masses[pdg]) + np.square(momentum))

def kinetic_energy(pdg: int, momentum: float) -> float:

    """ Return kinetic energy of particle from PDG number and momentum

    Args:
        pdg (int):        PDG code of particle
        momentum (float): momentum of particle (in GeV)

    Returns:
        float: kinetic energy of particle (in GeV)
    """
    return total_energy(pdg, momentum) - particle_masses[pdg]

# ---------------------------------------------------------------------------- #
#             All the stuff needed for ALEPH dE/dx parametrisation             #
# ---------------------------------------------------------------------------- #

p1 = 3.30
p2 = 8.80
p3 = 0.27
p4 = 0.75
p5 = 0.82

def beta_momentum(p: float,
                  m: float):
    
    """ Relativistic beta factor from momentum and mass

    Args:
        p (float): momentum of particle
        m (float): mass of particle

    Returns:
        float: beta factor
    """
    return (p/m)/np.sqrt(1+np.square(p/m))

def gamma_momentum(p, m):

    """ 

    Args:
        p (_type_): _description_
        m (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.sqrt(1+np.square(p/m))

def aleph_param_momentum(x, m, p1, p2, p3, p4, p5):
    return p1*(p2-np.power(beta_momentum(x, m), p4)-np.log(p3+1/np.power(beta_momentum(x, m)*gamma_momentum(x, m), p5)))/np.power(beta_momentum(x, m), p4)

def aleph_default(x, m):
    return aleph_param_momentum(x, m, p1, p2, p3, p4, p5)