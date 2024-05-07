import numpy as np
import uproot
import awkward
import pickle

from rich.progress import track

from garanapy import util
from garanapy import plotting

import inspect
from typing import List, Tuple, Callable
from pathlib import Path

class Neutrino:
    def __init__(self, e: awkward.highlevel.Record) -> None:
        self.type = e.NeutrinoType[0]
        self.cc   = bool(e.CCNC[0]-1)
        self.energy = np.sqrt(np.square(e.MCnuPX)+np.square(e.MCnuPY)+np.square(e.MCnuPZ))[0]
        self.position = np.array([e.MCVertexX[0], e.MCVertexY[0], e.MCVertexZ[0]])
        self.contained = util.in_fiducial(self.position)

    def __str__(self) -> str:
        return (f"    Type:   {self.type}\n"
                f"    Energy: {self.energy} GeV"
               )

    def __repr__(self) -> str:
        return str(self)
    
class MCParticle:
    def __init__(self, e: awkward.highlevel.Record, idx: int) -> None:
        self.id = idx
        self.pdg = e.GPartPdg[idx]
        self.status = e.GPartStatus[idx]

        self.energy = e.GPartE[idx]
        self.mass   = e.GPartMass[idx]

    def __str__(self) -> str:
        return (f"    PDG:    {self.pdg}\n"
                f"    Status: {self.status}"
               )

    def __repr__(self) -> str:
        return str(self)
    
class RecoParticle:
    def __init__(self, e: awkward.highlevel.Record, idx: int) -> None:
        self.id = idx

        self.mc_pdg      = e.MCPPDG[idx]
        self.mc_primary  = bool(e.MCPPrimary[idx])
        self.mc_momentum = e.MCPMomentumStart[idx]

        self.momentum = e.RecoMomentum[idx]

        self.Ecalo             = e.RecoTotalCaloEnergy[idx]
        self.dEdx              = e.RecoMeanCaloEnergy[idx]
        self.proton_dEdx_score = e.RecoProtonCaloScore[idx]

        self.Eecal      = e.RecoTotalECALEnergy[idx]
        self.Necal      = e.RecoNHitsECAL[idx]
        self.Emuid      = e.RecoTotalMuIDEnergy[idx]
        self.Nmuid      = e.RecoNHitsMuID[idx]
        self.muon_score = e.RecoMuonScore[idx]

        self.ecaled_end = e.RecoTrackEndECALed[idx]

        self.tof_beta         = e.RecoECALToFBeta[idx]
        self.proton_tof_score = e.RecoProtonToFScore[idx]

        self.charge = e.RecoCharge[idx]

        self.vertexed_end = e.RecoTrackEndVertexed[idx]

        self.track_start_x = e.TrackStartX[idx]
        self.track_start_y = e.TrackStartY[idx]
        self.track_start_z = e.TrackStartZ[idx]

        self.track_end_x = e.TrackEndX[idx]
        self.track_end_y = e.TrackEndY[idx]
        self.track_end_z = e.TrackEndZ[idx]

    def set_pid(self, pid) -> None:
        self.pid = pid

    def __str__(self) -> str:
        return (f"    Momentum:    {self.momentum} GeV"
               )

    def __repr__(self) -> str:
        return str(self)
    
class Event:
    def __init__(self, e: awkward.highlevel.Record, only_fsi: bool = True) -> None:
        self.nu = Neutrino(e)

        self.n_mcparticle = e.GPartPdg.layout.shape[0]
        self.mcparticle_list = []
        for i in range(self.n_mcparticle):
            if (e.GPartStatus[i] != 1) & only_fsi: continue
            self.mcparticle_list.append(MCParticle(e, i))

        self.n_recoparticle = e.RecoMomentum.layout.shape[0]
        self.recoparticle_list = []
        for i in range(self.n_recoparticle):
            self.recoparticle_list.append(RecoParticle(e, i))

        self.bad_direction = False
        self.set_direction()

        #self.has_muon = False
        #self.mc_primary_muon()

    def get_mcparticle(self, id: int) -> MCParticle:
        return self.mcparticle_list[id]
    
    def get_recoparticle(self, id: int) -> RecoParticle:
        return self.recoparticle_list[id]

    def __str__(self) -> str:
        return ("Neutrino:\n"+
                str(self.nu)
               )

    def __repr__(self) -> str:
        return str(self)
    
    # Temporary solution to get the start position of the reco particles into the Event
    def get_candidate_vertex(self) -> np.array:

        # Try first to get ECALed particles
        particles_ecaled   = [(p.id, p.Necal)    for p in self.recoparticle_list if p.ecaled_end != -1]
        # Try also to get vertexed particles
        particles_vertexed = [(p.id, p.momentum) for p in self.recoparticle_list if p.vertexed_end != -1]

        if (len(particles_ecaled) > 0):
            particles_ecaled = sorted(particles_ecaled, key=lambda x: x[1])
            ecaled_ref_particle = self.get_recoparticle(particles_ecaled[0][0])

            # If the track end ECALed is the End (0), then the true begin is the Begin (0)
            if(ecaled_ref_particle.ecaled_end == 0):
                vertex_candidate_pos = np.array([ecaled_ref_particle.track_start_x,
                                                 ecaled_ref_particle.track_start_y,
                                                 ecaled_ref_particle.track_start_z])
            # Else, if the end ECALed is the Begin (1), the true begin is the End (0)
            elif(ecaled_ref_particle.ecaled_end == 1):
                vertex_candidate_pos = np.array([ecaled_ref_particle.track_end_x,
                                                 ecaled_ref_particle.track_end_y,
                                                 ecaled_ref_particle.track_end_z])

        elif (len(particles_vertexed) > 0):
            particles_vertexed = sorted(particles_vertexed, key=lambda x: x[1])
            vertexed_ref_particle = self.get_recoparticle(particles_vertexed[0][0])

            # If the track end Vertexed is the Begin (1), then the true begin is the Begin (1)
            if(vertexed_ref_particle.vertexed_end == 1):
                vertex_candidate_pos = np.array([vertexed_ref_particle.track_start_x,
                                                 vertexed_ref_particle.track_start_y,
                                                 vertexed_ref_particle.track_start_z])
            # Else, if the end Vertexed is the End (0), the true begin is the End (0)
            elif(vertexed_ref_particle.vertexed_end == 0):
                vertex_candidate_pos = np.array([vertexed_ref_particle.track_end_x,
                                                 vertexed_ref_particle.track_end_y,
                                                 vertexed_ref_particle.track_end_z])

        else:
            vertex_candidate_pos = np.array([None,
                                             None,
                                             None])
            
        return vertex_candidate_pos
    
    def set_direction(self) -> None:

        vertex_candidate_pos = self.get_candidate_vertex()

        if (vertex_candidate_pos == None).all():
            self.bad_direction = True
            return

        for p in self.recoparticle_list:

            particle_begin = np.array([p.track_start_x,
                                       p.track_start_y,
                                       p.track_start_z])

            particle_end   = np.array([p.track_end_x,
                                       p.track_end_y,
                                       p.track_end_z])

            distance_begin = np.linalg.norm(particle_begin-vertex_candidate_pos)
            distance_end   = np.linalg.norm(particle_end-vertex_candidate_pos)

            if (distance_begin <= distance_end):
                p.start_x = particle_begin[0]
                p.start_y = particle_begin[1]
                p.start_z = particle_begin[2]
            else:
                p.start_x = particle_end[0]
                p.start_y = particle_end[1]
                p.start_z = particle_end[2]