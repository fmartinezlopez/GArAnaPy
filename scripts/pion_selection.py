import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from rich import print as rprint

from typing import Union

from garanapy import util
from garanapy import event
from garanapy import datamanager
from garanapy import plotting
from garanapy import idle

# ---------------------------------------------------------------------------- #
#                              Useful definitions                              #
# ---------------------------------------------------------------------------- #

# Masses of the particles (in GeV)
m_electron  =   0.511*1e-3
m_muon      = 105.658*1e-3
m_pion      = 139.570*1e-3
m_kaon      = 493.677*1e-3
m_proton    = 938.272*1e-3

particle_names = {11:   r"$e^{\pm}$",
                  13:   r"$\mu^{\pm}$",
                  211:  r"$\pi^{\pm}$",
                  321:  r"$K^{\pm}$",
                  2212: r"$p$"}

# ---------------------------------------------------------------------------- #
#             All the stuff needed for ALEPH dE/dx parametrisation             #
# ---------------------------------------------------------------------------- #

p1 = 3.30
p2 = 8.80
p3 = 0.27
p4 = 0.75
p5 = 0.82

def beta_momentum(x, m):
    return (x/m)/np.sqrt(1+np.square(x/m))

def gamma_momentum(x, m):
    return np.sqrt(1+np.square(x/m))

def aleph_param_momentum(x, m, p1, p2, p3, p4, p5):
    return p1*(p2-np.power(beta_momentum(x, m), p4)-np.log(p3+1/np.power(beta_momentum(x, m)*gamma_momentum(x, m), p5)))/np.power(beta_momentum(x, m), p4)

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def get_selected_numu(event: event.Event,
                      muon_cut: float):

    if (event.nu.type == 14)&(event.nu.cc == True)&(event.nu.contained)&(not event.bad_direction):
        # Create list of reco particles with negative charge and muon score greater than cut
        candidates = [(p.id, p.momentum, p.muon_score) for p in event.recoparticle_list if (p.charge == -1) & (p.muon_score >= muon_cut)]
        # Sort candidate list by increasing momentum
        candidates = sorted(candidates, key=lambda x: x[1])

        if (len(candidates) > 0):
            # Primary muon identified
            muon_id = candidates[0][0]
            return muon_id
        else:
            return -1
    else:
        return -1
    
# This function will return an iterable for each event instead of a single value
# The DataManager knows how to handle this when loading the corresponding Spectrum
def get_true_pion_ke(event: event.Event,
                     muon_cut: float,
                     p_thres: float):

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:
        return [mcp.energy-mcp.mass for mcp in event.mcparticle_list
                if (np.abs(mcp.pdg) == 211)
                &(np.sqrt(np.square(mcp.energy)-np.square(mcp.mass)) >= p_thres)]
    else:
        return None
    
def get_reco_pion_ke(event: event.Event,
                     muon_cut: float,
                     true_pdg: int,
                     p_min: float,
                     proton_calo_cut: float,
                     proton_tof_cut: float,
                     delta_calo: float,
                     distance_cut: float):

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:
        pion_ke = []
        muon_start = np.array([event.get_recoparticle(muon_id).start_x,
                               event.get_recoparticle(muon_id).start_y,
                               event.get_recoparticle(muon_id).start_z])

        for p in event.recoparticle_list:
            if p.id == muon_id: continue

            if true_pdg == -1:
                pass
            elif np.abs(p.mc_pdg) != true_pdg:
                continue

            reco_momentum = p.momentum
            if reco_momentum < p_min: continue

            # First select based on proton scores
            proton_calo_score = p.proton_dEdx_score
            proton_tof_score  = p.proton_tof_score

            if (proton_calo_score <= proton_calo_cut)&(proton_tof_score <= proton_tof_cut):
                # Then check if the dE/dx makes sense with the pion hypothesis
                reco_dEdx     = p.dEdx

                expt_calo = aleph_param_momentum(reco_momentum,
                                                 m_pion,
                                                 p1, p2, p3, p4, p5)
                
                start = np.array([p.start_x,
                                  p.start_y,
                                  p.start_z])
                
                distance_to_muon = np.linalg.norm(muon_start-start)

                if (reco_dEdx >= expt_calo*(1-delta_calo))&(reco_dEdx < expt_calo*(1+delta_calo))&(distance_to_muon <= distance_cut):
                    pion_ke.append(np.sqrt(np.square(reco_momentum)+np.square(m_pion)) - m_pion)
                    #pion_ke.append(reco_momentum - m_pion)
        
        return pion_ke
    else:
        return None
    

def count_pion_mult(event: event.Event,
                    muon_cut: float,
                    p_thres: float,
                    proton_calo_cut: float,
                    proton_tof_cut: float,
                    delta_calo: float,
                    distance_cut: float):
    
    true_pion_ke = get_true_pion_ke(event, muon_cut, p_thres)
    reco_pion_ke = get_reco_pion_ke(event, muon_cut, -1, p_thres, proton_calo_cut, proton_tof_cut, delta_calo, distance_cut)

    if true_pion_ke is None:
        n_true_pion = 0
    else:
        n_true_pion = len(true_pion_ke)

    if true_pion_ke is None:
        n_reco_pion = 0
    else:
        n_reco_pion = len(reco_pion_ke)

    return n_true_pion, n_reco_pion

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
    
    p_thres = 0.08 # assume 80 MeV/c pion detection threshold

    muon_score_cut = 0.5
    proton_dEdx_score_cut = 0.8
    proton_tof_score_cut = 0.8
    pion_dEdx_rejection = 0.1
    pion_vertex_distance_cut = 100.0 # in cm

    # Create logarithmic binning
    kinetic_energy_bins = plotting.Binning(5e-3, 1e2, 30, log=True)

    var_true_pion_ke  = datamanager.Variable(get_true_pion_ke, muon_score_cut,
                                                               0.0) # don't use pion threshold to plot true distribution
    
    spec_true_pion_ke = datamanager.MultiSpectrum(var_true_pion_ke, kinetic_energy_bins)
    data_manager.add_spectrum(spec_true_pion_ke, "true_pion_ke")

    spec_reco_pion_ke = {}
    for pdg in [11, 13, 211, 321, 2212]:
        var_reco_pion_ke  = datamanager.Variable(get_reco_pion_ke, muon_score_cut,
                                                                   pdg,
                                                                   p_thres,
                                                                   proton_dEdx_score_cut,
                                                                   proton_tof_score_cut,
                                                                   pion_dEdx_rejection,
                                                                   pion_vertex_distance_cut)
        
        spec_reco_pion_ke[pdg] = datamanager.MultiSpectrum(var_reco_pion_ke, kinetic_energy_bins)
        data_manager.add_spectrum(spec_reco_pion_ke[pdg], f'reco_pion_ke_{pdg}')

    #multiplicity_bins = plotting.Binning(-0.5, 3.5, 4)
    multiplicity_bins = plotting.Binning(-0.5, 1.5, 2)
    multiplicity_bins.add_bin(100.)

    var_pion_mult = datamanager.Variable(count_pion_mult, muon_score_cut,
                                                          p_thres, 
                                                          proton_dEdx_score_cut,
                                                          proton_tof_score_cut,
                                                          pion_dEdx_rejection,
                                                          pion_vertex_distance_cut)
    
    spec_pion_mult = datamanager.Spectrum2D(var_pion_mult, multiplicity_bins, multiplicity_bins)
    data_manager.add_spectrum(spec_pion_mult, "pion_mult")

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    fig, ax = plt.subplots(figsize=(7,5))

    hist_true_pion_ke = spec_true_pion_ke.get_histogram()
    hist_true_pion_ke.set_label("GENIE")
    hist_true_pion_ke.plot_histogram_errorbar(ax, color="k", zorder=100)

    hist_reco_pion_ke = {}
    for pdg in [11, 13, 211, 321, 2212]:
        hist_reco_pion_ke[pdg] = spec_reco_pion_ke[pdg].get_histogram()
        hist_reco_pion_ke[pdg].set_label(particle_names[pdg])

    plotting.stack_histograms(hist_reco_pion_ke, ax)

    ax.set_xscale("log")
    ax.set_xlabel("Pion Kinetic Energy [GeV]", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper right")

    plt.savefig("numu_cc_pion_ke.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    #label_pos = np.arange(0.0, 4.5, 1.0)
    #labels = [str(i)+r"$\pi^{\pm}$" for i in range(4)]
    #labels.append(r"$\geq$ 4 $\pi^{\pm}$")
    label_pos = np.arange(0.0, 2.5, 1.0)
    labels = [str(i)+r"$\pi^{\pm}$" for i in range(2)]
    labels.append(r"$\geq$ 2 $\pi^{\pm}$")

    hist_pion_mult = spec_pion_mult.get_histogram()
    hist_pion_mult.plot_histogram(fig,
                                  ax,
                                  cmap="inferno",
                                  vprob=True,
                                  col_norm=True,
                                  row_norm=False,
                                  annotations=True,
                                  scale=True,
                                  colorbar=False,
                                  annotation_size=20)

    ax.set_xticks(label_pos)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticks(label_pos)
    ax.set_yticklabels(labels, rotation=90, va='center', fontsize=20)

    ax.set_xlabel(r"GENIE $\pi^{\pm}$ Multiplicity", fontsize=20, labelpad=10, loc="right")
    ax.set_ylabel(r"Reco $\pi^{\pm}$ Multiplicity", fontsize=20, labelpad=10, loc="top")

    plt.savefig("numu_cc_pion_multiplicity.pdf", dpi=500, bbox_inches='tight')

    if batch:
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    total = [sum(hist.counts) for key, hist in hist_reco_pion_ke.items() if key != 211]
    total_total = sum([sum(hist.counts) for key, hist in hist_reco_pion_ke.items()]) # lol
    labels_pos = np.arange(0, 4, 1)
    labels = [r"$e^{\pm}$", r"$\mu^{\pm}$", r"$K^{\pm}$", r"$p$"]

    for i in range(4):
        plt.bar(labels_pos[i], 100*total[i]/total_total)

    ax.set_xticks(labels_pos)
    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylabel("Percentage", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()

    plt.savefig("numu_cc_pion_misids.pdf", dpi=500, bbox_inches='tight')
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