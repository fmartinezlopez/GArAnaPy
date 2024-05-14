import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from colorspacious import cspace_converter

from typing import Union

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

def get_nice_param(param, label, units="", newline=True):
    value = param.value
    error = param.stderr

    error_str = '%s' % float('%.2g' % error)
    error_str = str(error_str)
    error_exp_pos = error_str.find("e")
    if (error_exp_pos != -1):
        error_exp = int(error_str[error_exp_pos+1:])
    
    s = "".join(re.findall(r'\d+', error_str))
    if (len([i for i in range(len(s)) if not s.startswith("0", i)]) == 1):
        error_str = error_str+"0"

    exponent = False
    value_str = str(value)
    value_exp_pos = value_str.find("e")
    if (value_exp_pos != -1):
        exponent = True
        value_exp = int(value_str[value_exp_pos+1:])
        error_str = '%s' % float('%.2g' % (float(error_str[:error_exp_pos])*np.power(10.0, (error_exp-value_exp))))


    value_str = value_str[:value_str.find(".")]+value_str[value_str.find("."):][:len(error_str)-1]

    if exponent:
        f'{value_str} '+r"\pm"+f' {error_str}'

    if newline:
        return label+f'{value_str} '+r"\pm"+f' {error_str}'+units+f'\n'
    else:
        return label+f'{value_str} '+r"\pm"+f' {error_str}'+units

def plot_fit_summary(ax, results, x=0.05, y=0.95, name_dict=None):

    props = dict(boxstyle='square',
                 edgecolor="red",
                 facecolor="white",
                 alpha=1.0,
                 linewidth=0.0)

    textstr = r"\begin{eqnarray*} "+r"\chi^{2}/ndf&=& "+f'{"{:.3f}".format(results.chisqr)}/{results.nfree}'+"\\\\"
    
    for param in [*results.params.values()]:
        if param.vary:
            
            name = param.name
            if name_dict is not None:
                name = name_dict[name]
            
            textstr += get_nice_param(param, name+r" &=&", units="\\\\", newline=False)
        
    textstr += r"\end{eqnarray*}"

    ax.text(x, y, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left', bbox=props, fontdict=mono)

class Binning:
    def __init__(self, xmin:   Union[float, None] = None,
                       xmax:   Union[float, None] = None,
                       nbins:  Union[int, None] = None,
                       log:    bool = False,
                       custom: Union[list, None] = None) -> None:
        
        if custom is not None:
            self.bins = custom
            self.xmin = custom[0]
            self.xmax = custom[-1]
            self.nbins = len(custom)-1
        else:
            self.xmin  = xmin
            self.xmax  = xmax
            self.nbins = nbins
            self.log   = log

            self.make_bins()

    def make_bins(self) -> None:
        if self.log:
            self.bins = np.logspace(np.log10(self.xmin),
                                    np.log10(self.xmax),
                                    self.nbins+1)
        else:
            self.bins = np.linspace(self.xmin,
                                    self.xmax,
                                    self.nbins+1)
    
    def add_bin(self, bin) -> None:
        self.bins = np.append(self.bins, bin)


# LBL and ND analyses binnings

kBinEdges    = [0.,  0.5,  1.,  1.25, 1.5, 1.75,
                2.,  2.25, 2.5, 2.75, 3.,  3.25,
                3.5, 3.75, 4.,  5.,   6.,  10.]
kYBinEdges   = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]

kV3BinEdges  = [0.,  0.75, 1.,  1.25, 1.5, 1.75, 2., 2.25,
                2.5, 2.75, 3.,  3.25, 3.5, 3.75, 4., 4.25,
                4.5, 5.,   5.5, 6.,   7.,  8.,   10.]
kYV3BinEdges = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

kHadBinEdges = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 
                0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 
                1.5, 2., 20.]
kLepBinEdges = [0., 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25,
                2.375, 2.5, 2.75, 3., 3.25, 3.5, 4., 20]


kFDRecoBinning         = Binning(custom=kBinEdges)
kNDRecoBinning         = Binning(custom=kBinEdges)
kHadRecoBinning        = Binning(custom=kHadBinEdges)
kLepRecoBinning        = Binning(custom=kLepBinEdges)
kFDRecoV3Binning       = Binning(custom=kV3BinEdges)
kNDRecoV3Binning       = Binning(custom=kV3BinEdges)
kNDRecoOABinning       = Binning(0.0, 4.0, 20)
kYBinning              = Binning(custom=kYBinEdges)
kYV3Binning            = Binning(custom=kYV3BinEdges)
kTrueBinning           = Binning(0.0, 10.0, 100)
kTrueCoarseBinning     = Binning(0.0, 10.0, 20)
kRecoCoarseBinning     = Binning(0.0, 10.0, 20)
kRecoVeryCoarseBinning = Binning(0.0, 10.0, 5)
kOneBinBinning         = Binning(0.0, 10.0, 1)

class Histogram:
    def __init__(self,
                 binning: Binning,
                 counts = None,
                 label = None) -> None:
        
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

def histogram_ratio(a_hist: Histogram,
                    another_hist: Histogram) -> Histogram:
    
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

class Histogram2D:
    def __init__(self,
                 binning_x: Binning,
                 binning_y: Binning,
                 counts = None) -> None:
        
        self.binning_x = binning_x
        self.binning_y = binning_y

        if counts is not None:
            try:
                iterator = iter(counts)
            except TypeError:
                self.counts = np.ones((self.binning_x.nbins-1, self.binning_y.nbins-1))*counts
            else:
                self.counts = counts
        else:
            self.counts = np.empty((self.binning_x.nbins-1, self.binning_y.nbins-1))

    def make_hist(self, data_x, data_y) -> None:
        counts, _, _ = np.histogram2d(data_x,
                                      data_y,
                                      bins=[self.binning_x.bins, self.binning_y.bins])
        self.counts = counts
        self.yerr = np.sqrt(counts)

    def plot_histogram(self,
                       fig,
                       ax=None,
                       cmap="inferno",
                       vprob=False,
                       col_norm=False,
                       row_norm=False,
                       annotations=False,
                       scale=False,
                       colorbar=False,
                       **kwargs) -> None:
        if ax is None:
            fig, ax = plt.subplots()

        entries = self.counts.copy()

        if col_norm:
            for k in range(self.binning_x.nbins):
                entries[k] = entries[k]/np.sum(entries[k])
        elif row_norm:
            for k in range(self.binning_y.nbins):
                entries[:,k] = entries[:,k]/np.sum(entries[:,k])

        if vprob:
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        else:
            norm = matplotlib.colors.Normalize()

        cmap = matplotlib.cm.get_cmap(cmap)

        if scale:
            ax.imshow(entries.T,
                      origin='lower',
                      cmap=cmap,
                      norm=norm,
                      **kwargs)
        else:
            X, Y = np.meshgrid(self.binning_x.bins, self.binning_y.bins)
            ax.pcolormesh(X, Y, entries.T,
                          cmap=cmap,
                          norm=norm,
                          rasterized=True,
                          **kwargs)

        if colorbar:
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        if annotations:
            for m in range(self.binning_x.nbins):
                for n in range(self.binning_y.nbins):
                    # Little trick so all annotations can be visible
                    rgb = cmap(norm(entries[m, n]))[:1]
                    light = cspace_converter("sRGB1", "CAM02-UCS")(rgb)[0]
                    tcolor = "k" if light >= 75 else "w"
                    ax.text(m, n, "{:.2f}".format(entries[m, n]),
                                ha="center", va="center", color=tcolor,
                                fontsize=12)