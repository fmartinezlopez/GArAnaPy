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

class Histogram:
    def __init__(self, bins, counts = None, label = None) -> None:
        self.bins = bins

        if counts is not None:
            self.counts = counts
        else:
            self.counts = []

        if label is not None:
            self.label = label
        else:
            self.label = ""

        self.data = []

        self.update = False

    def set_label(self, label) -> None:
        self.label = label

    def set_yerr(self, yerr) -> None:
        self.yerr = yerr

    def add_count(self, count) -> None:
        self.counts.append(count)

    def add_data(self, data) -> None:
        self.data.append(data)
        self.update = True

    def make_hist(self) -> None:
        counts, _ = np.histogram(self.data, self.bins)
        self.counts = counts
        self.yerr = np.sqrt(counts)

    def make_hist_from_input(self, data) -> None:
        counts, _ = np.histogram(data, self.bins)
        self.counts = counts
        self.yerr = np.sqrt(counts)

    def get_histogram(self, ax=None, **kwargs) -> None:
        if self.update:
            self.make_hist()
            self.update = False

        if ax is None:
            fig, ax = plt.subplots()

        ax.stairs(self.counts, self.bins, linewidth=1.5, label=self.label, **kwargs)

    def get_histogram_errorbar(self, ax=None, yerr=True, **kwargs) -> None:
        if self.update:
            self.make_hist()
            self.update = False

        if ax is None:
            fig, ax = plt.subplots()

        if yerr:
            _yerr = self.yerr
        else:
            _yerr = 0.

        ax.errorbar(bin_centres(self.bins), self.counts, xerr=np.diff(self.bins) / 2, yerr=_yerr, linestyle="None", elinewidth=1.5, label=self.label, **kwargs)

def histogram_ratio(a_hist: Histogram, another_hist: Histogram) -> Histogram:
    if (a_hist.bins != another_hist.bins).all():
        raise ValueError("Histograms have different binnings!")
    
    ratio = Histogram(a_hist.bins, a_hist.counts/another_hist.counts)
    ratio.set_yerr(ratio.counts*np.sqrt(1/a_hist.counts+1/another_hist.counts))
    
    return ratio