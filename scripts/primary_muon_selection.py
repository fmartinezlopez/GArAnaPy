import numpy as np
import matplotlib.pyplot as plt

from rich import print as rprint

import click
import pickle

from garanapy import util
from garanapy import event
from garanapy import datamanager
from garanapy import plotting
from garanapy import idle

# ---------------------------------------------------------------------------- #
#                              Useful definitions                              #
# ---------------------------------------------------------------------------- #

particle_names = {11:   r"$e^{\pm}$",
                  13:   r"$\mu^{\pm}$",
                  211:  r"$\pi^{\pm}$",
                  321:  r"$K^{\pm}$",
                  2212: r"$p$"}

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def get_selected_muon_momentum(event: event.Event,
                               nu_pdg: int,
                               cc: bool,
                               muon_cut: float,
                               pdg: int):
    
    """ Return reconstructed momentum of selected primary muon candidate

    Args:
        event (event.Event): event to process
        nu_pdg (int):        true neutrino PDG code to select
        cc (bool):           CC interaction?
        muon_cut (float):    value of muon score to use for numu CC selection
        pdg (int): _description_

    Returns:
        _type_: _description_
    """

    if (event.nu.type == nu_pdg)&(event.nu.cc == cc)&(event.nu.contained):
        # Create list of reco particles with negative charge and muon score greater than cut
        candidates = [(p.id, p.momentum, p.muon_score) for p in event.recoparticle_list if (p.charge == -1) & (p.muon_score >= muon_cut)]
        # Sort candidate list by increasing momentum
        candidates = sorted(candidates, key=lambda x: x[1])

        if (len(candidates) > 0):
            # Primary muon identified
            muon_candidate = event.get_recoparticle(candidates[0][0])
            if (muon_candidate.mc_pdg == pdg):
                return muon_candidate.momentum
            else:
                return None
        else:
            return None
    else:
        return None
    
def get_selected_true_muon_momentum(event: event.Event,
                                    nu_pdg: int,
                                    cc: bool,
                                    muon_cut: float):
    
    """ Return true momentum of primary muon

    Args:
        event (event.Event): event to process
        nu_pdg (int):        true neutrino PDG code to select
        cc (bool):           CC interaction?
        muon_cut (float):    value of muon score to use for numu CC selection
        pdg (int): _description_

    Returns:
        _type_: _description_
    """

    if (event.nu.type == nu_pdg)&(event.nu.cc == cc)&(event.nu.contained):
        # Create list of reco particles with negative charge and muon score greater than cut
        candidates = [(p.id, p.momentum, p.muon_score) for p in event.recoparticle_list if (p.charge == -1) & (p.muon_score >= muon_cut)]
        # Sort candidate list by increasing momentum
        candidates = sorted(candidates, key=lambda x: x[1])

        if (len(candidates) > 0):
            # Primary muon identified
            p_max = -999.
            for mcp in event.mcparticle_list:
                mcp_momentum = np.sqrt(np.square(mcp.energy)-np.square(mcp.mass))
                if (mcp.pdg == 13)&(mcp_momentum > p_max):
                    p_max = mcp_momentum
            if p_max > 0.:
                return p_max
            else:
                return None
        else:
            return None
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

    # Create some binning
    momentum_bins = plotting.Binning(0.0, 8.0, 40)

    spec_selected_muon_momentum = {}
    for p in [11, 13, 211, 321, 2212]:
        var_selected_muon_momentum  = datamanager.Variable(get_selected_muon_momentum, 14, True, 0.5, p)
        spec_selected_muon_momentum[p] = datamanager.Spectrum(var_selected_muon_momentum, momentum_bins)
        data_manager.add_spectrum(spec_selected_muon_momentum[p], f'selected_muon_momentum_{p}')

    var_selected_true_muon_momentum  = datamanager.Variable(get_selected_true_muon_momentum, 14, True, 0.5)
    spec_selected_true_muon_momentum = datamanager.Spectrum(var_selected_true_muon_momentum, momentum_bins)
    data_manager.add_spectrum(spec_selected_true_muon_momentum, 'selected_true_muon_momentum')

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    rprint("\n[green]Plot, plot, plot\n")

    fig, ax = plt.subplots(figsize=(7,5))

    hist_selected_muon_momentum = {}
    for p in [11, 13, 211, 321, 2212]:
        hist_selected_muon_momentum[p] = spec_selected_muon_momentum[p].get_histogram()
        hist_selected_muon_momentum[p].set_label(particle_names[p])

    plotting.stack_histograms(hist_selected_muon_momentum, ax)

    hist_selected_true_muon_momentum = spec_selected_true_muon_momentum.get_histogram()
    hist_selected_true_muon_momentum.set_label("GENIE")
    hist_selected_true_muon_momentum.plot_histogram_errorbar(ax, color="black", zorder=101)

    ax.set_xlabel("Momentum [GeV/c]", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper right")

    plt.savefig("numu_cc_selection_muon_momentum.pdf", dpi=500, bbox_inches='tight')
    if batch:
        plt.close()
    else:
        plt.show()

    # Same thing but with a logarithmic binning
    log_momentum_bins = plotting.Binning(1e-1, 2e1, 40, log=True)

    fig, ax = plt.subplots(figsize=(7,6))

    hist_selected_muon_momentum = {}
    for p in [11, 13, 211, 321, 2212]:
        spec_selected_muon_momentum[p].set_binning(log_momentum_bins)
        hist_selected_muon_momentum[p] = spec_selected_muon_momentum[p].get_histogram()
        hist_selected_muon_momentum[p].set_label(particle_names[p])

    plotting.stack_histograms(hist_selected_muon_momentum, ax)

    spec_selected_true_muon_momentum.set_binning(log_momentum_bins)
    hist_selected_true_muon_momentum = spec_selected_true_muon_momentum.get_histogram()
    hist_selected_true_muon_momentum.set_label("GENIE")
    hist_selected_true_muon_momentum.plot_histogram_errorbar(ax, color="black", zorder=101)

    ax.set_xlabel(r"Momentum [GeV/$c$]", fontsize=24, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=24, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xscale("log")
    ax.grid()
    ax.legend(fontsize=20, loc="upper right")

    plotting.SimulationWIP(x=0.04, y=0.85, ax=ax, color="black", fontsize=20)

    plt.savefig("numu_cc_selection_muon_momentum_log.pdf", dpi=500, bbox_inches='tight')
    if batch:
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    total = [sum(hist.counts) for key, hist in hist_selected_muon_momentum.items() if key != 13]
    total_total = sum([sum(hist.counts) for key, hist in hist_selected_muon_momentum.items()]) # lol
    labels_pos = np.arange(0, 4, 1)
    labels = [r"$e^{\pm}$", r"$\pi^{\pm}$", r"$K^{\pm}$", r"$p$"]

    for i in range(4):
        plt.bar(labels_pos[i], 100*total[i]/total_total)

    ax.set_xticks(labels_pos)
    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylabel("Percentage", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()

    plt.savefig("numu_cc_selection_muon_misids.pdf", dpi=500, bbox_inches='tight')
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