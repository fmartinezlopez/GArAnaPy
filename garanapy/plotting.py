import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dune")
mono = {'family' : 'monospace'}
plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'

custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": [
        r"\usepackage{amsmath}", # for the align environment
        ],
    }
plt.rcParams.update(custom_preamble)

# todo: why is this not working?
color_cycle = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color_cycle)

def bin_centres(bins):
    """_summary_

    Args:
        bins (np.array, list): input bins

    Returns:
        np.array, list: centres of corresponding bins
    """
    return bins[:-1] + np.diff(bins) / 2

class Binning:
    def __init__(self, xmin: float, xmax: float, nbins: int, log: bool = False) -> None:
        self.xmin  = xmin
        self.xmax  = xmax
        self.nbins = nbins
        self.log   = log

        self.make_bins()

    def make_bins(self):
        if self.log:
            self.bins = np.logspace(np.log10(self.xmin),
                                    np.log10(self.xmax),
                                    self.nbins)
        else:
            self.bins = np.linspace(self.xmin,
                                    self.xmax,
                                    self.nbins)

class Histogram:
    def __init__(self, binning: Binning, counts = None, label = None) -> None:
        self.binning = binning

        if counts is not None:
            try:
                iterator = iter(counts)
            except TypeError:
                self.counts = np.ones(self.binning.nbins-1)*counts
            else:
                self.counts = counts
        else:
            self.counts = []

        if label is not None:
            self.label = label
        else:
            self.label = ""

    def set_label(self, label) -> None:
        self.label = label

    def set_yerr(self, yerr) -> None:
        self.yerr = yerr

    def add_count(self, count) -> None:
        self.counts.append(count)

    def make_hist(self, data) -> None:
        counts, _ = np.histogram(data, self.binning.bins)
        self.counts = counts
        self.yerr = np.sqrt(counts)

    def plot_histogram(self, ax=None, **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots()

        ax.stairs(self.counts, self.binning.bins, linewidth=1.5, label=self.label, **kwargs)

    def plot_histogram_errorbar(self, ax=None, yerr=True, **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots()

        if yerr:
            _yerr = self.yerr
        else:
            _yerr = 0.

        ax.errorbar(bin_centres(self.binning.bins), self.counts, xerr=np.diff(self.binning.bins) / 2, yerr=_yerr, linestyle="None", elinewidth=1.5, label=self.label, **kwargs)

def histogram_ratio(a_hist: Histogram, another_hist: Histogram) -> Histogram:
    if (a_hist.binning.bins != another_hist.binning.bins).all():
        raise ValueError("Histograms have different binnings!")
    
    ratio = Histogram(a_hist.binning, a_hist.counts/another_hist.counts)
    ratio.set_yerr(ratio.counts*np.sqrt(1/a_hist.counts+1/another_hist.counts))
    
    return ratio

def stack_histograms(hist_collection, ax):
    key_list = list(hist_collection.keys())
    idx = np.argsort([np.sum(hist_collection[key].counts) for key in hist_collection])

    stack =  Histogram(hist_collection[key_list[0]].binning, 0.)
    order = 10

    for i in idx:
        stack.counts += hist_collection[key_list[i]].counts
        stack.set_label(hist_collection[key_list[i]].label)
        stack.plot_histogram(ax, zorder=order, fill=True)
        order -= 1