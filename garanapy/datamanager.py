import numpy as np
import uproot
import pickle

import inspect
from typing import List, Callable, Union, Tuple
from pathlib import Path

from rich.progress import track

from garanapy import util
from garanapy import plotting
from garanapy import event

def create_event(e):
    return event.Event(e)

class Variable:
    def __init__(self,
                 func: Callable,
                 *args) -> None:

        first_arg_name = inspect.getfullargspec(func).args[0] # get name of first argument in func
        try:
            first_arg_type = func.__annotations__[first_arg_name]
            if (first_arg_type != event.Event):
                raise TypeError("First argument in input function must be of type event.Event!")
        except KeyError:
            raise KeyError("Input function for Variable object was not properly type hinted!")

        self.func  = func
        self.args  = args

    def get_wrapped_func(self,
                         e: event.Event):
        return self.func(e, *self.args)
    
class Spectrum:
    def __init__(self,
                 variable: Variable,
                 binning: plotting.Binning) -> None:
        
        self.variable = variable
        self.hist = plotting.Histogram(binning=binning)
        self.data = []

    def add_data(self,
                 data_point: Union[int, float]) -> None:
        
        self.data.append(data_point)

    def set_binning(self,
                    binning: plotting.Binning) -> None:
        
        self.hist = plotting.Histogram(binning=binning)

    def get_histogram(self) -> plotting.Histogram:

        self.hist.make_hist(self.data)
        return self.hist
    
class MultiSpectrum(Spectrum):
    def add_data(self,
                 data_points: List[float]) -> None:
        
        self.data.extend(data_points)
    
class Spectrum2D(Spectrum):
    def __init__(self,
                 variable: Variable,
                 binning_x: plotting.Binning,
                 binning_y: plotting.Binning) -> None:
        
        self.variable = variable
        self.hist = plotting.Histogram2D(binning_x=binning_x, binning_y=binning_y)
        self.data_x = []
        self.data_y= []

    def add_data(self,
                 data_point: Tuple[int, float]) -> None:
        
        self.data_x.append(data_point[0])
        self.data_y.append(data_point[1])

    def get_histogram(self) -> plotting.Histogram2D:
        
        self.hist.make_hist(self.data_x, self.data_y)
        return self.hist
    
class DataManager:
    def __init__(self) -> None:
        self.event_list = []
        self.spectrum_list = {}

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as output:  # Overwrites any existing file
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def open_events(self, data_path: str, tree_name: str = "recoparticlesana/AnaTree", n_files: int = -1) -> None:
        file_list = util.get_datafile_list(data_path)
        file_list = util.sorted_nicely(file_list)
        _n_files = len(file_list)

        if (n_files == -1) | (n_files > _n_files):
            n_files = _n_files

        for i in track(range(n_files), description="Reading data..."):
            file = uproot.open(Path(data_path, file_list[i]))
            tree = file.get(tree_name)
            self.event_list.extend(list(map(create_event, tree.arrays())))

    def add_spectrum(self, variable: Variable, key: str) -> None:
        self.spectrum_list[key] = variable

    def load_spectrum(self, spectrum: Spectrum) -> None:
        for e in self.event_list:
            ret = spectrum.variable.get_wrapped_func(e)
            if ret is not None:
                spectrum.add_data(ret)
    
    def load_spectra(self) -> None:
        for e in self.event_list:
            for _, spectrum in self.spectrum_list.items():
                ret = spectrum.variable.get_wrapped_func(e)
                if ret is not None:
                    spectrum.add_data(ret)