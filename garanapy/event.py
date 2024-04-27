import numpy as np
import uproot
import awkward

from rich.progress import track

from garanapy import util
from typing import List
from pathlib import Path

def create_event(e):
    return Event(e)

class Neutrino:
    def __init__(self, e: awkward.highlevel.Record) -> None:
        self.type = e.NeutrinoType[0]
        self.energy = np.sqrt(np.square(e.MCnuPX)+np.square(e.MCnuPY)+np.square(e.MCnuPZ))[0]

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

        self.has_muon = False

    def get_recoparticle(self, id: int) -> RecoParticle:
        return self.recoparticle_list[id]

    def search_primary_muon(self, cut: float) -> None:
        # Create list of reco particles with negative charge and muon score greater than cut
        candidates = [(p.id, p.momentum, p.muon_score) for p in self.recoparticle_list if (p.charge == -1) & (p.muon_score >= cut)]
        # Sort candidate list by increasing momentum
        candidates = sorted(candidates, key=lambda x: x[1])

        if (len(candidates) > 0):
            # Get the muon candidate using the ID of the highest momentum
            # reco particle in candidate list
            muon_id = candidates[0][0]
            muon = self.get_recoparticle(muon_id)

            # Set the PID of that reco particle to 13
            muon.set_pid(13)

            # Update 
            self.has_muon = True
            self.muon_id = muon_id

    def __str__(self) -> str:
        return ("Neutrino:\n"+
                str(self.nu)
               )

    def __repr__(self) -> str:
        return str(self)
    
class DataManager:
    def __init__(self) -> None:
        self.event_list = []
        self._true_nu_energy = False
        self._primary_muon_momentum = False

    def open_events(self, data_path: str, tree_name: str = "recoparticlesana/AnaTree", n_files: int = -1) -> None:
        file_list = util.get_datafile_list(data_path)
        file_list = util.sorted_nicely(file_list)

        if n_files == -1:
            n_files = len(file_list)

        for i in track(range(n_files), description="Reading data..."):
            file = uproot.open(Path(data_path, file_list[i]))
            tree = file.get(tree_name)
            self.event_list.extend(list(map(create_event, tree.arrays())))

    def get_true_nu_energy(self) -> List[float]:
        if self._true_nu_energy:
            return self.true_nu_energy
        else:
            self._true_nu_energy = True
            self.true_nu_energy = [e.nu.energy for e in self.event_list]
            return self.get_true_nu_energy()
        
    def search_primary_muon(self, cut: float = 0.5) -> None:
        for e in self.event_list:
            e.search_primary_muon(cut)

    def get_primary_muon_momentum(self) -> List[float]:
        if self._primary_muon_momentum:
            return self.primary_muon_momentum
        else:
            self._primary_muon_momentum = True
            self.primary_muon_momentum = [e.get_recoparticle(e.muon_id).momentum for e in self.event_list if e.has_muon]
            return self.get_primary_muon_momentum()