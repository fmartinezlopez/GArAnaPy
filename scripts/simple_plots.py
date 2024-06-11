import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from rich import print as rprint

from typing import Union, Tuple

from garanapy import util
from garanapy import event
from garanapy import datamanager
from garanapy import plotting
from garanapy import idle

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def get_muon_score(event: event.Event,
                   pdg: int) -> int:

    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        return [p.muon_score for p in event.recoparticle_list if (p.Necal > 0)&(p.mc_pdg == pdg)]
    else:
        return None

def get_proton_dEdx_score(event: event.Event,
                          pdg: int) -> int:

    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        return [p.proton_dEdx_score for p in event.recoparticle_list if (p.mc_pdg == pdg)&(p.momentum <= 1.5)]
    else:
        return None    

def get_proton_tof_score(event: event.Event,
                         pdg: int) -> int:

    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        return [p.proton_tof_score for p in event.recoparticle_list if (p.Necal > 0)&(p.mc_pdg == pdg)&(p.momentum >= 0.5)&(p.momentum < 3.0)]
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

    score_bins = plotting.Binning(0.0, 1.0, 50)

    spec_muon_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        var_muon_score  = datamanager.Variable(get_muon_score, pdg)
        spec_muon_score[pdg] = datamanager.MultiSpectrum(var_muon_score, score_bins)
        data_manager.add_spectrum(spec_muon_score[pdg], f'muon_score_{pdg}')

    spec_proton_dEdx_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        var_proton_dEdx_score  = datamanager.Variable(get_proton_dEdx_score, pdg)
        spec_proton_dEdx_score[pdg] = datamanager.MultiSpectrum(var_proton_dEdx_score, score_bins)
        data_manager.add_spectrum(spec_proton_dEdx_score[pdg], f'proton_dEdx_score_{pdg}')
    
    spec_proton_tof_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        var_proton_tof_score  = datamanager.Variable(get_proton_tof_score, pdg)
        spec_proton_tof_score[pdg] = datamanager.MultiSpectrum(var_proton_tof_score, score_bins)
        data_manager.add_spectrum(spec_proton_tof_score[pdg], f'proton_tof_score_{pdg}')

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    rprint("\n[green]Plot, plot, plot\n")

    fig, ax = plt.subplots(figsize=(7,5))

    hist_muon_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        hist_muon_score[pdg] = spec_muon_score[pdg].get_histogram()
        hist_muon_score[pdg].set_label(util.particle_names[pdg])

    plotting.stack_histograms(hist_muon_score, ax)

    ax.set_xlabel(r"Muon score", fontsize=14, labelpad=10, loc="right")
    ax.set_ylabel(r"Counts", fontsize=14, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid()

    plt.savefig("numu_cc_muon_scores.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    hist_proton_dEdx_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        hist_proton_dEdx_score[pdg] = spec_proton_dEdx_score[pdg].get_histogram()
        hist_proton_dEdx_score[pdg].set_label(util.particle_names[pdg])

    plotting.stack_histograms(hist_proton_dEdx_score, ax)

    ax.set_xlabel(r"Proton dE/dx score", fontsize=14, labelpad=10, loc="right")
    ax.set_ylabel(r"Counts", fontsize=14, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid()

    plt.savefig("numu_cc_proton_dEdx_scores.pdf", dpi=500, bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(7,5))

    hist_proton_tof_score = {}
    for pdg in [11, 13, 211, 321, 2212]:
        hist_proton_tof_score[pdg] = spec_proton_tof_score[pdg].get_histogram()
        hist_proton_tof_score[pdg].set_label(util.particle_names[pdg])

    plotting.stack_histograms(hist_proton_tof_score, ax)

    ax.set_xlabel(r"Proton ToF score", fontsize=14, labelpad=10, loc="right")
    ax.set_ylabel(r"Counts", fontsize=14, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    ax.grid()

    plt.savefig("numu_cc_proton_tof_scores.pdf", dpi=500, bbox_inches='tight')
    
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