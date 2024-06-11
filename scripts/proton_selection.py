import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from rich import print as rprint

from typing import Union, Tuple, List

from garanapy import util
from garanapy import event
from garanapy import datamanager
from garanapy import plotting
from garanapy import idle

from lmfit import Model

# ---------------------------------------------------------------------------- #
#                             Definitions for fits                             #
# ---------------------------------------------------------------------------- #

def gaussian(x, m, s, a):
    return a*np.exp(-np.square((x-m)/s)/2.) # don't add normalisation term!

model = Model(gaussian)

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def beta_momentum(x, m):
    return 1/np.sqrt(1+np.square(m/x))

def get_dEdx_bin(event: event.Event,
                 p_min: float,
                 p_max: float,
                 pdg: int) -> List[float]:
    """_summary_

    Args:
        event (event.Event): event to process
        p_min (float): _description_
        p_max (float): _description_
        pdg (int): True PID (-1 if you want all)

    Returns:
        List[float]: fractional residual
    """

    # Get true numu CC interactions
    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        ret = []
        for p in event.recoparticle_list:
            # Select only particle if it matches the desired true PID
            if (p.mc_pdg != pdg)&(pdg != -1): continue
            reco_momentum = p.momentum
            if (p.momentum >= p_min)&(p.momentum < p_max):
                # If particle is in the corresponding momentum bin compare dE/dx with that of protons
                expt_calo = util.aleph_default(reco_momentum, util.m_proton)
                ret.append((expt_calo-p.dEdx)/expt_calo)
        return ret
    else:
        return None
    
def get_beta_bin(event: event.Event,
                 p_min: float,
                 p_max: float,
                 pdg: int) -> List[float]:

    # Get true numu CC interactions
    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        ret = []
        for p in event.recoparticle_list:
            # Select only particle if it matches the desired true PID
            if (p.mc_pdg != pdg)&(pdg != -1): continue
            reco_momentum = p.momentum
            if (p.momentum >= p_min)&(p.momentum < p_max):
                # If particle is in the corresponding momentum bin compare ToF beta with that of protons
                expt_beta = beta_momentum(reco_momentum, util.m_proton)
                ret.append((expt_beta-p.tof_beta)/expt_beta)
        return ret
    else:
        return None
    
# ---------------------------------------------------------------------------- #
#                            Main function goes here                           #
# ---------------------------------------------------------------------------- #

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--input_type', type=click.Choice(["ROOT", "pickle"]),
              help="Select input file type", default='ROOT', show_default=True)
@click.option('-n', '--n_files', type=int,
              help="Number of input files to load", default=-1, show_default=True)
@click.option('-b', '--batch', is_flag=True,
              help="Run in batch mode, don't open plots", default=False, show_default=True)
@click.option('-i', '--interactive', is_flag=True,
              help="Open interactive terminal before finishing", default=False, show_default=True)
def cli(data_path: str, input_type: str, n_files: int, batch: bool, interactive: bool) -> None:

    # Start process (for benchmarking)
    process = idle.Process()
    process.start_process()

    # Create DataManager object and load data...
    if input_type == "ROOT":
        data_manager = datamanager.DataManager()
        data_manager.open_events(data_path, n_files=n_files)
    # ...or just load it from a .pickle file
    elif input_type == "pickle":
        data_manager = util.open_pickle_data(data_path)
    else:
        raise ValueError("Invalid input type!")
    
    # True PDG codes to compare
    pdg_list = [13, 211, 2212]

    # Momentum bin for dE/dx comparison
    p_min = 0.90
    p_max = 0.95

    dEdx_frac_bins = plotting.Binning(-0.3, 0.3, 50)

    var_dEdx_total  = datamanager.Variable(get_dEdx_bin, p_min, p_max, -1)
    spec_dEdx_total = datamanager.MultiSpectrum(var_dEdx_total, dEdx_frac_bins)
    data_manager.add_spectrum(spec_dEdx_total, "dEdx_total")

    spec_dEdx = {}
    for pdg in pdg_list:
        var_dEdx  = datamanager.Variable(get_dEdx_bin, p_min, p_max, pdg)
        spec_dEdx[pdg] = datamanager.MultiSpectrum(var_dEdx, dEdx_frac_bins)
        data_manager.add_spectrum(spec_dEdx[pdg], f'dEdx_{pdg}')

    # Momentum bin for ToF beta comparison
    p_min_tof = 1.45
    p_max_tof = 1.50

    tof_frac_bins = plotting.Binning(-0.4, 0.4, 50)

    var_beta_total  = datamanager.Variable(get_beta_bin, p_min_tof, p_max_tof, -1)
    spec_beta_total = datamanager.MultiSpectrum(var_beta_total, tof_frac_bins)
    data_manager.add_spectrum(spec_beta_total, "beta_total")

    spec_beta = {}
    for pdg in pdg_list:
        var_beta  = datamanager.Variable(get_beta_bin, p_min_tof, p_max_tof, pdg)
        spec_beta[pdg] = datamanager.MultiSpectrum(var_beta, tof_frac_bins)
        data_manager.add_spectrum(spec_beta[pdg], f'beta_{pdg}')

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    rprint("\n[green]Plot, plot, plot\n")

    fig = plt.figure(figsize=(12,5))
    gs = fig.add_gridspec(1, 2, wspace=0.1)
    axs = gs.subplots(sharex=False, sharey=True)

    hist_dEdx_total = spec_dEdx_total.get_histogram()
    hist_dEdx_total.set_label("")
    hist_dEdx_total.plot_histogram_errorbar(axs[0], color="k")

    hist_dEdx = {}
    total_fit = np.zeros(1001)
    x = np.linspace(dEdx_frac_bins.xmin, dEdx_frac_bins.xmax, 1001)
    for pdg in pdg_list:
        hist_dEdx[pdg] = spec_dEdx[pdg].get_histogram()

        # Generate the parameter list with some decent
        # guesses and constraints
        params = model.make_params(m = dict(value=0.0),
                                   s = dict(value=0.05, min=0.0001),
                                   a = dict(value=1000.0, min=0.0))

        # Fit the model
        results = model.fit(hist_dEdx[pdg].counts+1,
                            params,
                            x=plotting.bin_centres(dEdx_frac_bins.bins),
                            weights=1/np.sqrt(hist_dEdx[pdg].counts+1))

        fit_result = gaussian(x, *results.values.values())
        total_fit += fit_result
        axs[0].plot(x, fit_result, zorder=99, label=util.particle_names[pdg])

    axs[0].plot(x, total_fit, zorder=99, color="gray", linestyle="--", label="Total")

    axs[0].set_xlabel(r"$\frac{\mathrm{d}E/\mathrm{d}x_{proton}-\mathrm{d}E/\mathrm{d}x_{reco}}{\mathrm{d}E/\mathrm{d}x_{proton}}$", fontsize=20, labelpad=10, loc="right")
    axs[0].set_ylabel(r"Counts", fontsize=20, labelpad=10, loc="top")
    axs[0].tick_params(axis='both', which='major', labelsize=20)

    # You may want a log scale...
    #axs[0].set_yscale("log")
    #axs[0].set_ylim(5e-1, 4e3)
    axs[0].set_xticks([-0.30, -0.15, 0.00, 0.15, 0.30])
    axs[0].grid()

    # Add label with corresponding momentum bin
    plt.text(0.05, 0.70, r"${:.2f} \leq p_{{reco}} < {:.2f}$ GeV/$c$".format(p_min, p_max), fontsize=20, transform=axs[0].transAxes)
    # And DUNE Simulation Work in progress label
    plotting.SimulationWIP(x=0.05, y=0.80, ax=axs[0], color="black", fontsize=20)

    hist_beta_total = spec_beta_total.get_histogram()
    hist_beta_total.set_label("")
    hist_beta_total.plot_histogram_errorbar(axs[1], color="k")

    hist_beta = {}
    total_fit = np.zeros(1001)
    x = np.linspace(tof_frac_bins.xmin, tof_frac_bins.xmax, 1001)
    for pdg in pdg_list:
        hist_beta[pdg] = spec_beta[pdg].get_histogram()

        # Generate the parameter list with some decent
        # guesses and constraints
        params = model.make_params(m = dict(value=-0.15),
                                   s = dict(value=0.01, min=0.0001),
                                   a = dict(value=100.0, min=0.0))

        # Fit the model
        results = model.fit(hist_beta[pdg].counts+1,
                            params,
                            x=plotting.bin_centres(tof_frac_bins.bins),
                            weights=1/np.sqrt(hist_beta[pdg].counts+1))

        fit_result = gaussian(x, *results.values.values())
        total_fit += fit_result
        axs[1].plot(x, fit_result, zorder=99, label=util.particle_names[pdg])

    axs[1].plot(x, total_fit, zorder=99, color="gray", linestyle="--", label="Total")

    axs[1].set_xlabel(r"$\frac{\beta_{proton}-\beta_{reco}}{\beta_{proton}}$", fontsize=20, labelpad=10, loc="right")
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].legend(fontsize=20, loc="center right")

    # You may want a log scale...
    #axs[1].set_yscale("log")
    #axs[1].set_ylim(5e-1, 4e3)
    axs[1].set_xticks([-0.40, -0.20, 0.00, 0.20, 0.40])
    axs[1].grid()

    # Add label with corresponding momentum bin
    plt.text(0.05, 0.70, r"${:.2f} \leq p_{{reco}} < {:.2f}$ GeV/$c$".format(p_min_tof, p_max_tof), fontsize=20, transform=axs[1].transAxes)
    # And DUNE Simulation Work in progress label
    plotting.SimulationWIP(x=0.05, y=0.80, ax=axs[1], color="black", fontsize=20)

    plt.savefig("numu_cc_proton_selection_example_no_log.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    process.end_process()

    if interactive:
        import IPython
        IPython.embed(color="neutral")

if __name__ == "__main__":

    cli()