import numpy as np
import matplotlib.pyplot as plt

import click
import pickle

from typing import Union

from garanapy import util
from garanapy import event
from garanapy import datamanager
from garanapy import plotting

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
        pdg_total_energy = [13, 22, 111]
    else:
        pdg_kinec_energy = [211, 2212]
        pdg_total_energy = [13, 22, 111, 211]

    mc_erec = 0.0

    for p in event.mcparticle_list:

        abs_pdg = np.abs(p.pdg)
        energy = p.energy

        if abs_pdg in pdg_kinec_energy:
            mc_erec += energy-util.particle_masses[abs_pdg]

        elif abs_pdg in pdg_total_energy:
            mc_erec += energy

    return mc_erec
    
def get_true_bias(event: event.Event,
                  muon_cut: float,
                  pion_mass: bool) -> Union[float, None]:
    
    """ Return true energy bias for events passing the numu CC cuts

    Args:
        event (event.Event): event to process
        muon_cut (float):    value of muon score to use for numu CC selection
        pion_mass (bool):    add pion mass in reconstructed energy?

    Returns:
        Union[float, None]: true neutrino energy bias
    """

    muon_id = get_selected_numu(event, muon_cut)

    if muon_id > 0:

        mc_erec = get_mc_erec(event, pion_mass)

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
    
def get_reco_frac_bias(event: event.Event,
                       muon_cut: float,
                       pion_mass: bool,
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

        return (event.nu.energy-reco_erec)/event.nu.energy
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

    # Create DataManager object and load data...
    if input_type == "ROOT":
        data_manager = datamanager.DataManager()
        data_manager.open_events(data_path, n_files=n_files)
    # ...or just load it from a .pickle file
    elif input_type == "pickle":
        with open(data_path, 'rb') as input:
            data_manager = pickle.load(input)
    else:
        raise ValueError("Invalid input type!")

    # Energy bias binning (MeV)
    energy_bias_bins = plotting.Binning(-1000.0, 0.0, 50)

    var_true_bias  = datamanager.Variable(get_true_bias, 0.5, True)
    spec_true_bias = datamanager.Spectrum(var_true_bias, energy_bias_bins)
    data_manager.add_spectrum(spec_true_bias, "true_bias")

    var_true_bias_no_pion  = datamanager.Variable(get_true_bias, 0.5, False)
    spec_true_bias_no_pion = datamanager.Spectrum(var_true_bias_no_pion, energy_bias_bins)
    data_manager.add_spectrum(spec_true_bias_no_pion, "true_bias_no_pion")

    # For the reco, lets get the fractional bias: (True-Reco)/True
    # Create an appropriate binning for it
    energy_frac_bias_bins = plotting.Binning(-2.0, 2.0, 50)

    """""
    Default parameters for get_reco_frac_bias
    
        Inputs for numu CC selection:
    
            muon_cut        = 0.5    [score]
    
        Inputs for energy reconstruction:
    
            pion_mass       = True   [bool]
    
        Inputs for pion selection:
    
            proton_calo_cut = 0.8    [score]
            proton_tof_cut  = 0.8    [score]
            delta_calo      = 0.1    [%]
            distance_cut    = 100.   [cm]
    """""

    var_reco_frac_bias  = datamanager.Variable(get_reco_frac_bias, 0.5, True, 0.8, 0.8, 0.1, 100.)
    spec_reco_frac_bias = datamanager.Spectrum(var_reco_frac_bias, energy_frac_bias_bins)
    data_manager.add_spectrum(spec_reco_frac_bias, "reco_frac_bias")

    # Once all the spectra have been added we can load them
    data_manager.load_spectra()

    # Now the plotting bit

    # First we plot the bias in reconstructed neutrino energy
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
    plt.savefig("numu_cc_energy_bias.pdf", dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(7,5))

    hist_reco_frac_bias = spec_reco_frac_bias.get_histogram()
    hist_reco_frac_bias.plot_histogram(ax, color="dodgerblue", zorder=100)

    ax.set_xlabel(r"$\frac{E_{\nu}^{true} - E_{\nu}^{reco}}{E_{\nu}^{true}}$", fontsize=16, labelpad=10, loc="right")
    ax.set_ylabel("Counts", fontsize=16, labelpad=10, loc="top")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid()

    plt.savefig("numu_cc_reco_energy_frac_bias.pdf", dpi=500, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    cli()