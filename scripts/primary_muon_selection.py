import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from garanapy import event
from garanapy import datamanager
from garanapy import plotting

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
def cli(data_path: str, input_type: str, n_files: int) -> None:

    # Create DataManager object and load data
    if input_type == "ROOT":
        data_manager = datamanager.DataManager()
        data_manager.open_events(data_path, n_files=n_files)
    elif input_type == "pickle":
        with open(data_path, 'rb') as input:
            data_manager = pickle.load(input)
    else:
        raise ValueError("Invalid input type!")

    # Create some binning
    momentum_bins = plotting.Binning(0.0, 8.0, 40)

    spec_selected_muon_momentum = {}
    for p in [11, 13, 211, 321, 2212]:
        var_selected_muon_momentum  = datamanager.Variable(get_selected_muon_momentum, 14, True, 0.5, p)
        spec_selected_muon_momentum[p] = datamanager.Spectrum(var_selected_muon_momentum, momentum_bins)
        data_manager.add_spectrum(spec_selected_muon_momentum[p], f'selected_muon_momentum_{p}')

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    fig, ax = plt.subplots(figsize=(7,5))

    hist_selected_muon_momentum = {}
    for p in [11, 13, 211, 321, 2212]:
        hist_selected_muon_momentum[p] = spec_selected_muon_momentum[p].get_histogram()
        hist_selected_muon_momentum[p].set_label(particle_names[p])

    plotting.stack_histograms(hist_selected_muon_momentum, ax)

    ax.set_xlabel("Momentum [GeV/c]", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper right")

    plt.savefig("numu_cc_selection_muon_momentum.pdf", dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    total = [sum(hist.counts) for key, hist in  hist_selected_muon_momentum.items() if key != 13]
    labels_pos = np.arange(0, 4, 1)
    labels = [r"$e^{\pm}$", r"$\pi^{\pm}$", r"$K^{\pm}$", r"$p$"]

    for i in range(4):
        plt.bar(labels_pos[i], total[i])

    ax.set_xticks(labels_pos)
    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()

    plt.savefig("numu_cc_selection_muon_misids.pdf", dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    cli()