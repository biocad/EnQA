import re
from tempfile import TemporaryDirectory
from typing import Any, List, Set, Tuple
from pathlib import Path

import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import standard_aa_names, three_to_one
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO


class ClearTrash(PDB.Select):
    def accept_residue(self, residue):
        return residue._id[0] == " "


class SelectSubsequence(PDB.Select):
    def __init__(self, accepted: set) -> None:
        self.accepted = accepted
        
    def accept_residue(self, residue) -> Any:
        return residue._id in self.accepted


def get_chain_sequencies(structure: Structure) -> List[List[str]]:
    sequencies = []
    for protein in structure:
        sequencies.append([])
        for chain in protein:
            sequencies[-1].append("")
            for amino in chain:
                if amino.resname in standard_aa_names:
                    sequencies[-1][-1] += three_to_one(amino.resname)

    return sequencies

def find_common_subsequence(res_seq: str, model_seq: str) -> Tuple[int, int, int]:
    len_res, len_mod = len(res_seq), len(model_seq)
    dynamic = np.zeros((len_res + 1, len_mod + 1))

    for i in range(len_res):
        for j in range(len_mod):
            if res_seq[i] == model_seq[j]:
                dynamic[i + 1][j + 1] = dynamic[i][j] + 1

    i, j = np.unravel_index(np.argmax(dynamic), dynamic.shape)
    length = int(np.max(dynamic))

    return i - length, j - length, length

def get_accepted_indexes_in_chain(
    initial_structure: Structure, indexes: Tuple[int, int]
    ) -> Set[Any]:
    start, end = indexes
    position = 0

    accepted = set()
    for protein in initial_structure:
        for chain in protein:
            for amino in chain:
                if position >= start and position < end:
                    accepted.add(amino._id)
                position += 1
    return accepted

def reindex_structure(structure: Structure) -> None:
    index = 0
    for model in structure:
        for caine in model:
            for residue in caine:
                residue._id = (" ", index, " ")
                index += 1

def get_clear_structures(struct: Structure) -> Structure:
    io = PDB.PDBIO()
    parser = PDB.PDBParser(QUIET=True)
    
    with TemporaryDirectory() as tmpdirname:
        io.set_structure(struct)
        io.save(tmpdirname + "/clear.pdb", ClearTrash())
        struct = parser.get_structure("", tmpdirname + "/clear.pdb")
        
        reindex_structure(struct)

    return struct

def get_cropped_structure(struct: PDB.Structure, accepted_idxs: Set) -> PDB.Structure:
    io = PDB.PDBIO()
    parser = PDB.PDBParser(QUIET=True)

    with TemporaryDirectory() as tmpdirname:
        io.set_structure(struct)
        io.save(tmpdirname + "/consistent.pdb", SelectSubsequence(accepted_idxs))
        struct = parser.get_structure("", tmpdirname + "/consistent.pdb")
        reindex_structure(struct)

    return struct

def initialize_structure_by_chains(chains: List[PDB.Chain.Chain]) -> Structure:
    new_struct = PDB.StructureBuilder.StructureBuilder()
    new_struct.init_structure("")
    new_struct.init_model("")
    new_struct = new_struct.get_structure()
    for idx, chain in enumerate(chains):
        chain._id = "ABCDEFG"[idx]
        new_struct[""].add(chain)
    return new_struct

def make_structures_consistent(
    first_structure: Structure, 
    second_structure: Structure
) -> Tuple[Structure, Structure]:

    real_complex, docked_complex = map(get_clear_structures, [first_structure, second_structure])

    real_sequencies = get_chain_sequencies(real_complex)[0]
    docked_sequencies = get_chain_sequencies(docked_complex)[0]

    real_complex_chains = list(real_complex.get_chains())
    docked_complex_chains = list(docked_complex.get_chains())

    real_order = []
    docked_order = []
    ordered_real_sequencies = []
    ordered_docked_sequencies = []
    intersect_length = []
    real_starts = []
    docked_starts = []

    max_intersect_length = 0
    best_match = tuple()
    real_and_docked_starts = tuple()
    for idx_doc, doc_seq in enumerate(docked_sequencies):
        max_intersect_length = 0
        best_match = tuple()
        real_and_docked_starts = tuple()
        for idx_real, real_seq in enumerate(real_sequencies):
            doc_start, real_start, length = find_common_subsequence(doc_seq, real_seq)
            if length > max_intersect_length:
                max_intersect_length = length
                best_match = (idx_real, idx_doc)
                real_and_docked_starts = (real_start, doc_start)
        real_starts.append(real_and_docked_starts[0])
        docked_starts.append(real_and_docked_starts[1])
        real_order.append(best_match[0])
        docked_order.append(best_match[1])
        ordered_real_sequencies.append(real_sequencies[best_match[0]])
        ordered_docked_sequencies.append(docked_sequencies[best_match[1]])
        intersect_length.append(max_intersect_length)
    
    real_complex_chains_cropped = []
    docked_complex_chains_cropped = []

    for idx_real, idx_docked, length, rl_strt, dcd_strt in zip(
        real_order, 
        docked_order, 
        intersect_length, 
        real_starts, 
        docked_starts):

        real_chn, dockd_chn = real_complex_chains[idx_real], docked_complex_chains[idx_docked]

        real_chn = initialize_structure_by_chains([real_chn])
        dockd_chn = initialize_structure_by_chains([dockd_chn])

        accepted_idxs_real = get_accepted_indexes_in_chain(real_chn, (rl_strt, rl_strt + length))
        accepted_idxs_docked = get_accepted_indexes_in_chain(
            dockd_chn, 
            (dcd_strt, dcd_strt + length)
        )
        real_chn = get_cropped_structure(real_chn, accepted_idxs_real)
        dockd_chn = get_cropped_structure(dockd_chn, accepted_idxs_docked)
        real_complex_chains_cropped.extend(real_chn.get_chains())
        docked_complex_chains_cropped.extend(dockd_chn.get_chains())
    
    real_consistent_complex = initialize_structure_by_chains(real_complex_chains_cropped)
    docked_consistent_complex = initialize_structure_by_chains(docked_complex_chains_cropped)

    return real_consistent_complex, docked_consistent_complex

def join_chains(chains: List[PDB.Chain.Chain], 
                new_name: str) -> List[PDB.Chain.Chain]:
    first_chain_len = 1000
    for _ in chains[0]:
        first_chain_len += 1    
    residue_counter = 0
    for chn in chains[1:]:
        for residue in chn:
            residue_counter += 1
            old_id = list(residue.id)
            old_id[1] = first_chain_len + residue_counter
            residue.id = tuple(old_id)
            chains[0].add(residue)
    chains[0]._id = new_name
    return chains[0]

def join_chains_and_initialize_structure(
    chains: List[PDB.Chain.Chain], 
    initial_len: int = 10000
) -> Structure:
    first_chain_len = initial_len
    for _ in chains[0]:
        first_chain_len += 1    
    residue_counter = 0
    for chn in chains[1:]:
        for residue in chn:
            residue_counter += 1
            old_id = list(residue.id)
            old_id[1] = first_chain_len + residue_counter
            residue.id = tuple(old_id)
            chains[0].add(residue)
    
    new_struct = PDB.StructureBuilder.StructureBuilder()
    new_struct.init_structure("")
    new_struct.init_model("")
    new_struct = new_struct.get_structure()
    chains[0]._id = "A"
    new_struct[""].add(chains[0])

    return new_struct

def init_structure(chains: List[PDB.Chain.Chain]) -> Structure:
    new_struct = PDB.StructureBuilder.StructureBuilder()
    new_struct.init_structure("")
    new_struct.init_model("")
    new_struct = new_struct.get_structure()
    for chain in chains:
        new_struct[""].add(chain)
    
    return new_struct

def merge_chains(chains_to_merge_1: List[PDB.Chain.Chain], 
                 chains_to_merge_2: List[PDB.Chain.Chain], 
                 path_to_output: Path):
    """
    Here we merge chains of antibody and write it to temporary PDB file
    """
    merged_1 = join_chains(chains_to_merge_1, new_name="A")
    merged_2 = join_chains(chains_to_merge_2, new_name="B")
    struct = init_structure([merged_1, merged_2])
    io=PDBIO()
    io.set_structure(struct)
    io.save(path_to_output)
