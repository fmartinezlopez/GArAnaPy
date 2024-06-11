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

from lmfit import Model

# ---------------------------------------------------------------------------- #
#                             Definitions for fits                             #
# ---------------------------------------------------------------------------- #

def gaussian(x, m, s, a):
    return a*np.exp(-np.square((x-m)/s)/2.) # don't add normalisation term!

def double_gaussian(x, m1, s1, a1, m2, s2, a2):
    return gaussian(x, m1, s1, a1) + gaussian(x, m2, s2, a2)

model = Model(double_gaussian)

# ---------------------------------------------------------------------------- #
#      Place here the functions that will retrieve the data for each event     #
# ---------------------------------------------------------------------------- #

def get_selected_numu(event: event.Event,
                      muon_cut: float) -> int:
    
    """ Returns ID of reconstructed primary muon candidate
        or -1 if no candidate is found

    Args:
        event (event.Event): event to process
        muon_cut (float):    value of muon score to use for numu CC selection

    Returns:
        int: particle ID of primary muon candidate 
    """

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
    
def get_mc_erec(event: event.Event,
                pion_mass: bool) -> float:

    """ Return reconstructed neutrino energy using true GENIE MCParticles

    Args:
        event (event.Event): event to process
        pion_mass (bool):    add pion mass in reconstructed energy?

    Returns:
        float: "true" reconstructed neutrino energy (in GeV)
    """

    # Do you want to add the pion mass to the energy?
    if pion_mass:
        pdg_kinec_energy = [2212]
        #pdg_total_energy = [13, 22, 111, 211]
        pdg_total_energy = [13, 211]
    else:
        pdg_kinec_energy = [211, 2212]
        #pdg_total_energy = [13, 22, 111]
        pdg_total_energy = [13]

    mc_erec = 0.0

    for p in event.mcparticle_list:

        abs_pdg = np.abs(p.pdg)
        energy = p.energy

        if abs_pdg in pdg_kinec_energy:
            mc_erec += energy-util.particle_masses[abs_pdg]

        elif abs_pdg in pdg_total_energy:
            mc_erec += energy

    return mc_erec

def get_enu_vs_mc_erec(event: event.Event,
                       muon_cut: float,
                       pion_mass: bool):
    
    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:

        mc_erec = get_mc_erec(event, pion_mass)

        return event.nu.energy, mc_erec
    
    else:
        return None
    
def get_true_bias(event: event.Event,
                  muon_cut: float,
                  pion_mass: bool,
                  frac_bias: bool) -> Union[float, None]:
    
    """ Return true energy (maybe fractional) bias for events passing the numu CC cuts

    Args:
        event (event.Event): event to process
        muon_cut (float):    value of muon score to use for numu CC selection
        pion_mass (bool):    add pion mass in reconstructed energy?
        frac_bias (bool):    compute fractional bias?

    Returns:
        Union[float, None]: true neutrino energy bias
    """

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:

        mc_erec = get_mc_erec(event, pion_mass)

        if frac_bias:
            return (event.nu.energy-mc_erec)/event.nu.energy
        else:
            return (mc_erec-event.nu.energy)*1e3 # in MeV
    else:
        return None
    
def process_reco_particle(p: event.RecoParticle,
                          pion_mass: bool,
                          muon_id: int,
                          proton_calo_cut: float,
                          proton_tof_cut: float,
                          delta_calo: float,
                          muon_start: np.array,
                          distance_cut: float) -> float:

    """ Returns the contribution to the reconstructed neutrino energy
        of the given RecoParticle

    Args:
        p (event.RecoParticle):  reconstructed particle to process
        pion_mass (bool):        add charged pion mass in nu energy?
        muon_id (int):           particle ID of primary muon candidate
        proton_calo_cut (float): value of proton dE/dx score to use for pion selection
        proton_tof_cut (float):  value of proton ToF score to use for pion selection
        delta_calo (float):      allow % deviation around pion dE/dx ALEPH prediction
        muon_start (np.array):   start position of muon candidate
        distance_cut (float):    max distance from primary vertex to select primary pion (in cm)

    Returns:
        float: reconstructed energy contribution (in GeV)
    """

    id = p.id
    momentum = p.momentum

    # If this is the primary muon, return its total energy
    if (id == muon_id):
        return util.total_energy(13, momentum)
    
    # Lets say a particle is contained if both start and end point are inside FV
    # AND the particle doesn't have any ECAL clusters associated
    track_start_pos = np.array([p.track_start_x,
                                p.track_start_y,
                                p.track_start_z])

    track_end_pos   = np.array([p.track_end_x,
                                p.track_end_y,
                                p.track_end_z])
    
    contained = util.in_fiducial(track_start_pos)&util.in_fiducial(track_end_pos)&(p.Necal==0)

    # If the particle is contained, return the total energy deposited in the TPC
    # It's stored in keV, convert to GeV
    if contained:
        return p.Ecalo*1e-6
    
    # Check if the particle is a proton by checking the dE/dx and ToF values
    proton_calo_score = p.proton_dEdx_score
    proton_tof_score  = p.proton_tof_score
    
    if (proton_calo_score >= proton_calo_cut)|(proton_tof_score >= proton_tof_cut):
        return util.kinetic_energy(2212, momentum)
    
    # Select the charged pions using the ALEPH dE/dx prediction...
    reco_dEdx     = p.dEdx
    expt_calo = util.aleph_default(momentum,
                                   util.m_pion)
    
    # ...and cut additional backgrounds using distance to primary vertex
    start_pos = np.array([p.start_x,
                          p.start_y,
                          p.start_z])
    
    distance_to_muon = np.linalg.norm(muon_start-start_pos)

    if (reco_dEdx >= expt_calo*(1-delta_calo))&(reco_dEdx < expt_calo*(1+delta_calo))&(distance_to_muon <= distance_cut):
        # Choose whether to add the pion mass or not
        if pion_mass:
            return util.total_energy(211, momentum)
        else:
            return util.kinetic_energy(211, momentum)
    
    # If everything else fails just say it was an electron
    # Should this be only a kinetic contribution?
    return util.total_energy(11, momentum)

def get_reco_erec(event: event.Event,
                  pion_mass: bool,
                  muon_id: int,
                  proton_calo_cut: float,
                  proton_tof_cut: float,
                  delta_calo: float,
                  distance_cut: float) -> float:

    """ Returns the reconstructed neutrino energy for the given event

    Args:
        event (event.Event):     event to process
        pion_mass (bool):        add charged pion mass in nu energy?
        proton_calo_cut (float): value of proton dE/dx score to use for pion selection
        proton_tof_cut (float):  value of proton ToF score to use for pion selection
        delta_calo (float):      allow % deviation around pion dE/dx ALEPH prediction
        distance_cut (float):    max distance from primary vertex to select primary pion (in cm)

    Returns:
        float: reconstructed neutrino energy (in GeV)
    """
    
    reco_erec = 0.0

    muon_start = np.array([event.get_recoparticle(muon_id).start_x,
                           event.get_recoparticle(muon_id).start_y,
                           event.get_recoparticle(muon_id).start_z])

    for p in event.recoparticle_list:

        reco_erec += process_reco_particle(p,
                                           pion_mass,
                                           muon_id,
                                           proton_calo_cut,
                                           proton_tof_cut,
                                           delta_calo,
                                           muon_start,
                                           distance_cut)

    return reco_erec
    
def get_reco_bias(event: event.Event,
                  muon_cut: float,
                  pion_mass: bool,
                  frac_bias: bool,
                  proton_calo_cut: float,
                  proton_tof_cut: float,
                  delta_calo: float,
                  distance_cut: float) -> Union[float, None]:
    
    """ Returns the reconstructed nu energy fractional bias,
        i.e. (E_true - E_reco)/E_true, for each selected event

    Args:
        event (event.Event):     event to process
        muon_cut (float):        value of muon score to use for numu CC selection
        pion_mass (bool):        add charged pion mass in nu energy?
        frac_bias (bool):        compute fractional bias?
        proton_calo_cut (float): value of proton dE/dx score to use for pion selection
        proton_tof_cut (float):  value of proton ToF score to use for pion selection
        delta_calo (float):      allow % deviation around pion dE/dx ALEPH prediction
        distance_cut (float):    max distance from primary vertex to select primary pion (in cm)

    Returns:
        Union[float, None]:      neutrino energy fractional bias
    """

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:

        reco_erec = get_reco_erec(event,
                                  pion_mass,
                                  muon_id,
                                  proton_calo_cut,
                                  proton_tof_cut,
                                  delta_calo,
                                  distance_cut)

        if frac_bias:
            return (event.nu.energy-reco_erec)/event.nu.energy
        else:
            return (reco_erec-event.nu.energy)*1e3 # in MeV
    else:
        return None
    
def get_bias(event: event.Event,
             muon_cut: float,
             pion_mass: bool,
             frac_bias: bool,
             proton_calo_cut: float,
             proton_tof_cut: float,
             delta_calo: float,
             distance_cut: float) -> Union[float, None]:
    
    """ Returns the difference between the predictions of
        E_rec using MCParticles or RecoParticles

    Args:
        event (event.Event):     event to process
        muon_cut (float):        value of muon score to use for numu CC selection
        pion_mass (bool):        add charged pion mass in nu energy?
        frac_bias (bool):        compute fractional bias?
        proton_calo_cut (float): value of proton dE/dx score to use for pion selection
        proton_tof_cut (float):  value of proton ToF score to use for pion selection
        delta_calo (float):      allow % deviation around pion dE/dx ALEPH prediction
        distance_cut (float):    max distance from primary vertex to select primary pion (in cm)

    Returns:
        Union[float, None]:      neutrino energy fractional bias
    """

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:

        reco_erec = get_reco_erec(event,
                                  pion_mass,
                                  muon_id,
                                  proton_calo_cut,
                                  proton_tof_cut,
                                  delta_calo,
                                  distance_cut)
        
        mc_erec = get_mc_erec(event,
                              pion_mass)

        if frac_bias:
            return (mc_erec-reco_erec)/mc_erec
        else:
            return (reco_erec-mc_erec)*1e3 # in MeV
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

    var_enu_vs_mc_erec = datamanager.Variable(get_enu_vs_mc_erec, 0.5, True)
    spec_enu_vs_mc_erec = datamanager.Spectrum2D(var_enu_vs_mc_erec, plotting.kNDRecoBinning, plotting.kNDRecoBinning)
    data_manager.add_spectrum(spec_enu_vs_mc_erec, "enu_vs_mc_erec")

    # Energy bias binning (MeV)
    energy_bias_bins = plotting.Binning(-1000.0, 0.0, 50)

    # Get the (absolute) energy bias using the MCParticle collection
    var_true_bias  = datamanager.Variable(get_true_bias, 0.5, True, False)
    spec_true_bias = datamanager.Spectrum(var_true_bias, energy_bias_bins)
    data_manager.add_spectrum(spec_true_bias, "true_bias")

    var_true_bias_no_pion  = datamanager.Variable(get_true_bias, 0.5, False, False)
    spec_true_bias_no_pion = datamanager.Spectrum(var_true_bias_no_pion, energy_bias_bins)
    data_manager.add_spectrum(spec_true_bias_no_pion, "true_bias_no_pion")

    # Get the (absolute) energy bias using the RecoParticle collection
    var_reco_bias  = datamanager.Variable(get_reco_bias, 0.5, True, False, 0.8, 0.8, 0.1, 100.)
    spec_reco_bias = datamanager.Spectrum(var_reco_bias, energy_bias_bins)
    data_manager.add_spectrum(spec_reco_bias, "reco_bias")

    var_reco_bias_no_pion  = datamanager.Variable(get_reco_bias, 0.5, False, False, 0.8, 0.8, 0.1, 100.)
    spec_reco_bias_no_pion = datamanager.Spectrum(var_reco_bias_no_pion, energy_bias_bins)
    data_manager.add_spectrum(spec_reco_bias_no_pion, "reco_bias_no_pion")

    # For the reco, lets get the fractional bias: (True-Reco)/True
    # Create an appropriate binning for it
    energy_frac_bias_bins = plotting.Binning(-2.0, 2.0, 50)

    """""
    Default parameters for get_reco_bias
    
        Inputs for numu CC selection:
    
            muon_cut        = 0.5    [score]
    
        Inputs for energy reconstruction:
    
            pion_mass       = True   [bool]
            frac_bias       = True   [bool]
    
        Inputs for pion selection:
    
            proton_calo_cut = 0.8    [score]
            proton_tof_cut  = 0.8    [score]
            delta_calo      = 0.1    [%]
            distance_cut    = 100.   [cm]
    """""

    var_reco_frac_bias  = datamanager.Variable(get_reco_bias, 0.5, True, True, 0.8, 0.8, 0.1, 100.)
    spec_reco_frac_bias = datamanager.Spectrum(var_reco_frac_bias, energy_frac_bias_bins)
    data_manager.add_spectrum(spec_reco_frac_bias, "reco_frac_bias")

    var_true_frac_bias  = datamanager.Variable(get_true_bias, 0.5, True, True)
    spec_true_frac_bias = datamanager.Spectrum(var_true_frac_bias, energy_frac_bias_bins)
    data_manager.add_spectrum(spec_true_frac_bias, "true_frac_bias")

    # Get also the (fractional) difference between the two E_rec predictions

    other_energy_frac_bias_bins = plotting.Binning(-0.5, 0.5, 31)

    var_frac_bias  = datamanager.Variable(get_bias, 0.5, True, True, 0.8, 0.8, 0.1, 100.)
    spec_frac_bias = datamanager.Spectrum(var_frac_bias, other_energy_frac_bias_bins)
    data_manager.add_spectrum(spec_frac_bias, "frac_bias")

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit
    rprint("\n[green]Plot, plot, plot\n")

    # Plot 2D histogram showing the true neutrino energy versus
    # the reconstructed energy using the MCParticles list (adding
    # the pion mass)
    fig, ax = plt.subplots(figsize=(7,5))

    hist_enu_vs_mc_erec = spec_enu_vs_mc_erec.get_histogram()
    hist_enu_vs_mc_erec.plot_histogram(fig, ax,
                                       cmap="Greys",
                                       col_norm=True,
                                       vprob=True,
                                       colorbar=True)

    ax.set_xlabel(r"$E_{\nu}$ [GeV]", fontsize=14, labelpad=10, loc="right")
    ax.set_ylabel(r"$E_{rec}^{MC}$ [GeV]", fontsize=14, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig("numu_cc_enu_vs_mc_erec_recobinning.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    # Using a different binning, plot true nu energy versus MC erec
    # Also, plot it column normalised (migration matrix!)
    fig, ax = plt.subplots(figsize=(7,5))

    # Change to a uniform binning
    energy_bins = plotting.Binning(0.0, 8.0, 20)
    spec_enu_vs_mc_erec.set_binning(energy_bins, energy_bins)

    hist_enu_vs_mc_erec = spec_enu_vs_mc_erec.get_histogram()
    hist_enu_vs_mc_erec.plot_histogram(fig, ax,
                                       cmap="Greys",
                                       col_norm=True,
                                       vprob=True,
                                       colorbar=True)

    ax.set_xlabel(r"$E_{\nu}$ [GeV]", fontsize=14, labelpad=10, loc="right")
    ax.set_ylabel(r"$E_{rec}^{MC}$ [GeV]", fontsize=14, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig("numu_cc_enu_vs_mc_erec.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    # Plot the bias in reconstructed neutrino energy
    # obtained using the MCParticles from the selected events
    fig, ax = plt.subplots(figsize=(7,5))

    # We plot the distribution including the pion mass...
    hist_true_bias = spec_true_bias.get_histogram()
    hist_true_bias.set_label("Default")
    hist_true_bias.plot_histogram_errorbar(ax, color="b", zorder=100)

    # ...and also the distribution considering only the kinetic contribution
    hist_true_bias_no_pion = spec_true_bias_no_pion.get_histogram()
    hist_true_bias_no_pion.set_label(r"No $\pi^{\pm}$ mass")
    hist_true_bias_no_pion.plot_histogram(ax, color="r", zorder=99)

    # Put labels, grid, legend and nice tick parameters
    ax.set_xlabel("Energy bias [MeV]", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper left")

    # Save and show
    plt.savefig("numu_cc_energy_bias_true.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    # Same but for the RecoParticles from the selected events
    fig, ax = plt.subplots(figsize=(7,5))

    # We plot the distribution including the pion mass...
    hist_reco_bias = spec_reco_bias.get_histogram()
    hist_reco_bias.set_label("Default")
    hist_reco_bias.plot_histogram_errorbar(ax, color="b", zorder=100)

    # ...and also the distribution considering only the kinetic contribution
    hist_reco_bias_no_pion = spec_reco_bias_no_pion.get_histogram()
    hist_reco_bias_no_pion.set_label(r"No $\pi^{\pm}$ mass")
    hist_reco_bias_no_pion.plot_histogram(ax, color="r", zorder=99)

    # Put labels, grid, legend and nice tick parameters
    ax.set_xlabel("Energy bias [MeV]", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper left")

    # Save the plot
    plt.savefig("numu_cc_energy_bias_reco.pdf", dpi=500, bbox_inches='tight')

    if batch:
        plt.close()
    else:
        plt.show()

    # Plot the fractional energy bias, i.e. (E_nu - E_rec)/E_nu
    # obtained for both the MCParticle and RecoParticle cases
    fig, ax = plt.subplots(figsize=(7,5))

    hist_reco_frac_bias = spec_reco_frac_bias.get_histogram()
    hist_reco_frac_bias.set_label("Reco")
    hist_reco_frac_bias.plot_histogram(ax, color="dodgerblue", zorder=100)

    hist_true_frac_bias = spec_true_frac_bias.get_histogram()
    hist_true_frac_bias.set_label("MC")
    hist_true_frac_bias.plot_histogram_errorbar(ax, color="black", zorder=101)

    ax.set_xlabel(r"$\frac{E_{\nu} - E_{rec}}{E_{\nu}}$", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper left")

    plt.savefig("numu_cc_energy_frac_bias.pdf", dpi=500, bbox_inches='tight')
    
    if batch:
        plt.close()
    else:
        plt.show()

    # Plot the fractional bias between the E_rec predictions,
    # i.e. (E_rec^MC - E_rec)/E_rec^MC, fitting a double gaussian
    # to the resulting distribution
    fig, ax = plt.subplots(figsize=(7,5))

    hist_frac_bias = spec_frac_bias.get_histogram()
    hist_frac_bias.set_label("Data")

    # Generate the parameter list with some decent
    # guesses and constraints
    params = model.make_params(m1 = dict(value=0.0, vary=False),
                               s1 = dict(value=0.05, min=0.0001),
                               a1 = dict(value=1000.0, min=0.0),
                               m2 = dict(value=0.0, vary=False),
                               s2 = dict(value=0.1, min=0.0001),
                               a2 = dict(value=100.0, min=0.0))
    
    # Create a name dictionary for the variables for nice plotting ;)
    name_dict = {"m1": r"\mu_{core}",
                 "s1": r"\sigma_{core}",
                 "a1": r"A_{core}",
                 "m2": r"\mu_{tail}",
                 "s2": r"\sigma_{tail}",
                 "a2": r"A_{tail}"}

    # Fit the model
    results = model.fit(hist_frac_bias.counts+1,
                        params,
                        x=plotting.bin_centres(other_energy_frac_bias_bins.bins),
                        weights=1/np.sqrt(hist_frac_bias.counts+1))

    hist_frac_bias.plot_histogram_errorbar(ax, color="black", marker="o", markersize=3., zorder=100)

    x = np.linspace(other_energy_frac_bias_bins.xmin, other_energy_frac_bias_bins.xmax, 1001)

    ax.plot(x, double_gaussian(x, *results.values.values()), color="red", zorder=99, label="Fit")

    plotting.plot_fit_summary(ax, results, x=0.05, y=0.95, name_dict=name_dict)

    ax.set_xlabel(r"$\frac{E_{rec}^{MC} - E_{rec}}{E_{rec}^{MC}}$", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()
    ax.legend(fontsize=14, loc="upper right")

    plt.savefig("numu_cc_erec_frac_bias.pdf", dpi=500, bbox_inches='tight')
    
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