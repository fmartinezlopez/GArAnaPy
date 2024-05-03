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
        self.contained = util.points_in_cylinder(util.pt1_fid, util.pt2_fid, util.TPCFidRadius, self.position)

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

        self.tof_beta         = e.RecoECALToFBeta[idx]
        self.proton_tof_score = e.RecoProtonToFScore[idx]

        self.charge = e.RecoCharge[idx]

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