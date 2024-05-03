import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from garanapy import event
from garanapy import datamanager
from garanapy import plotting

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def get_all_numu_energy(event: event.Event, nu_pdg: int, cc: bool):
    # Select numu CC events using MC truth
    if (event.nu.type == nu_pdg)&(event.nu.cc == cc)&(event.nu.contained):
        return event.nu.energy
    else:
        return None

def get_selected_numu_energy(event: event.Event, nu_pdg: int, cc: bool, muon_cut: float):

    if (event.nu.type == nu_pdg)&(event.nu.cc == cc)&(event.nu.contained):
        # Create list of reco particles with negative charge and muon score greater than cut
        candidates = [(p.id, p.momentum, p.muon_score) for p in event.recoparticle_list if (p.charge == -1) & (p.muon_score >= muon_cut)]
        # Sort candidate list by increasing momentum
        candidates = sorted(candidates, key=lambda x: x[1])

        if (len(candidates) > 0):
            # Primary muon identified, return true nu energy
            return event.nu.energy
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
    energy_bins = np.linspace(0.0, 8.0, 40)

    # For each distribution you want to extract, you need to create a Variable object...
    var_all_numu_energy  = datamanager.Variable(get_all_numu_energy, 14, True)
    # ...which is then used to create a Spectrum with the desired binning...
    spec_all_numu_energy = datamanager.Spectrum(var_all_numu_energy, energy_bins)
    # ...and finally it is added to the DataManager to load it later
    data_manager.add_spectrum(spec_all_numu_energy, "all_numu_energy")

    # Repeat the same procedure for any other variables
    var_selected_numu_energy  = datamanager.Variable(get_selected_numu_energy, 14, True, 0.5)
    spec_selected_numu_energy = datamanager.Spectrum(var_selected_numu_energy, energy_bins)
    data_manager.add_spectrum(spec_selected_numu_energy, "selected_numu_energy")

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(2, 1, hspace=0.05, height_ratios=[3,1])
    axs = gs.subplots(sharex=True, sharey=False)

    # Each of the spectra now have a filled Histogram object
    # You can retrieve it and use the build-in get_histogram
    # method to plot it
    hist_all_numu_energy = spec_all_numu_energy.hist
    hist_all_numu_energy.set_label(r"True $\nu_{\mu}$ CC")
    hist_all_numu_energy.get_histogram_errorbar(axs[0], color="blue")

    hist_selected_numu_energy = spec_selected_numu_energy.hist
    hist_selected_numu_energy.set_label(r"Selected $\nu_{\mu}$ CC")
    hist_selected_numu_energy.get_histogram(axs[0], color="red")

    axs[0].set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].grid()
    axs[0].legend(fontsize=14, loc="upper right")

    # Ratios of two histograms can be created easily with histogram_ratio
    ratio_histogram = plotting.histogram_ratio(hist_selected_numu_energy, hist_all_numu_energy)
    ratio_histogram.get_histogram_errorbar(axs[1], color="black")

    axs[1].set_xlabel("Neutrino energy [GeV]", fontsize=16, labelpad=10, loc="right")
    axs[1].set_ylabel(r"$\varepsilon$", fontsize=16, labelpad=10, loc="top")
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].grid()
    axs[1].set_ylim(-0.05, 1.05)

    plt.savefig("numu_cc_selection_e_true.pdf", dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    cli()