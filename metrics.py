import os
from pathlib import Path
import shutil
import pandas as pd
from pdb2sql import StructureSimilarity
from pdb2sql.superpose import superpose
from Bio.PDB.PDBParser import PDBParser
from pdb2sql import interface
from pdb_utils_crank import merge_chains
import torch

def get_chains_to_merge(path_to_structure, ab_chain_names):
    chains_to_merge_1 = []
    chains_to_merge_2 = []
    parser = PDBParser()
    struct = parser.get_structure("structure", path_to_structure)
    for chain in struct[0].get_chains():
        if chain.id in ab_chain_names:
            chains_to_merge_1.append(chain)
        else:
            chains_to_merge_2.append(chain)
    if chains_to_merge_1:
        return chains_to_merge_1, chains_to_merge_2
    ab_chain_names=['A','B']
    chains_to_merge_2 = []
    for chain in struct[0].get_chains():
        if chain.id in ab_chain_names:
            chains_to_merge_1.append(chain)
        else:
            chains_to_merge_2.append(chain)
    return chains_to_merge_1, chains_to_merge_2

def parse_chains(comp_name):
    _, chains, _ = comp_name.split('_')
    ab_chains, ag_chains = chains.split('-')
    ab_chains_list = ab_chains.split('+')
    ag_chains_list = ag_chains.split('+')
    return ab_chains_list, ag_chains_list

def calculate_fnat(path_to_docked_structure, path_to_reference_structure, ab_chains):
    #here we calculate FNAT with pdb2sql
    decoy = path_to_docked_structure
    ref = path_to_reference_structure
    
    ab_docked, ag_docked = get_chains_to_merge(path_to_docked_structure, ab_chains)
    ab_ref, ag_ref = get_chains_to_merge(path_to_reference_structure, ab_chains)
    merge_chains(ab_docked, ag_docked, "docked_temp.pdb")
    merge_chains(ab_ref, ag_ref, "ref_temp.pdb")
    decoy = "docked_temp.pdb"
    ref = "ref_temp.pdb"
    # get here structure with neccessary format

    sim = StructureSimilarity(decoy,ref)
    fnat = sim.compute_fnat_pdb2sql()
    return fnat

def calculate_rmsd(path_to_docked_structure, path_to_reference_structure, ab_chains):
    ab_docked, ag_docked = get_chains_to_merge(path_to_docked_structure, ab_chains)
    ab_ref, ag_ref = get_chains_to_merge(path_to_reference_structure, ab_chains)
    parent_path = path_to_docked_structure.parent
    merge_chains(ab_docked, ag_docked, str(parent_path / "docked_temp.pdb"))
    merge_chains(ab_ref, ag_ref, str(parent_path / "ref_temp.pdb"))
    decoy = str(parent_path / "docked_temp.pdb")
    ref = str(parent_path / "ref_temp.pdb")
    

    sim = StructureSimilarity(decoy,ref)
    lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None, method='svd')
    irmsd = sim.compute_irmsd_pdb2sql()
    
    return irmsd, lrmsd

def get_interface(path_to_structure: Path, threshold: float = 8.5):
    """
    Get interface residues of a complex
    """
    interface_db = interface(str(path_to_structure))
    contacts = interface_db.get_contact_residues(cutoff=threshold)
    return contacts


def get_raw_mapping(path_to_complex_dir,joined_name='joined',joined_path='joined_real.pdb',temp_name='real_temp',temp_path='ref_temp.pdb'):
    mapping_res = dict()
    
    parser = PDBParser()
    struct_joined = parser.get_structure(joined_name, path_to_complex_dir / joined_path)
    struct_temp = parser.get_structure(temp_name, temp_path)
    res_joined = list()
    for chn in struct_joined[0]:
        for res in chn:
            res_joined.append((chn.id, res.id[1], res.resname))

    res_temp = list()
    for chn in struct_temp[0]:
        for res in chn:
            res_temp.append((chn.id, res.id[1], res.resname))
            
    assert len(res_temp) == len(res_joined)
            
    for res1, res2 in zip(res_joined, res_temp):
        mapping_res[res2] = res1
    
    return mapping_res

def read_lddt(path: Path):
    return pd.read_table(path, sep='\t', skiprows=10)

def get_values_from_lddt_results(path_to_complex, mapping, contacts,lddt_list=None):
    lddt = read_lddt(path_to_complex / "lddt.csv")
    if lddt_list is not None:
        lddt['Score']=lddt_list.item()
    aa_to_lddt = dict()
    for index, row in lddt.iterrows():
        aa_to_lddt[(row['Chain'], row['ResNum'], row['ResName'])] = row['Score']
    
    all_contacts = []
    for v in contacts.values():
        all_contacts += v

    lddts = []
#     print(mapping)
    for aa in all_contacts:
        mapped = mapping[aa]
        lddts.append(aa_to_lddt[mapped])
    return lddts
    

def get_lddt_by_interface_and_irmsd(path_to_complex_dir, threshold=6.5):
    ab_chains = ['A', 'B']
    irmsd, lrmsd = calculate_rmsd(path_to_complex_dir / "docked.pdb", path_to_complex_dir / "real.pdb", ab_chains)
    contacts = get_interface(path_to_complex_dir / "ref_temp.pdb", threshold=threshold)
    mapping = get_raw_mapping(path_to_complex_dir)
    lddts = get_values_from_lddt_results(path_to_complex_dir, mapping, contacts)
    
    mean_lddt = sum(lddts) / len(lddts)
    
    return mean_lddt, irmsd, lrmsd

def get_lddt_by_interface_and_irmsd_no_hardcode(path_to_complex_dir, threshold=6.5):
    ab_chains = ['A', 'B']
    irmsd, lrmsd = calculate_rmsd(path_to_complex_dir / "docked.pdb", path_to_complex_dir / "real.pdb", ab_chains)
    contacts = get_interface(path_to_complex_dir / "ref_temp.pdb", threshold=threshold)
    mapping = get_raw_mapping(path_to_complex_dir)
    lddts = get_values_from_lddt_results(path_to_complex_dir, mapping, contacts)
    
    mean_lddt = sum(lddts) / len(lddts)
    
    return mean_lddt, irmsd, lrmsd

def calculate_dockq():
    #TBA
    pass

def get_values_from_lddt_predictions(path_to_complex, mapping, contacts,lddt_list):
    lddt = read_lddt(path_to_complex / "lddt.csv")
    aa_to_lddt = dict()
    for index, row in lddt.iterrows():
        aa_to_lddt[(row['Chain'], row['ResNum'], row['ResName'])] = index
    all_contacts = []
    for v in contacts.values():
        all_contacts += v
    lddts = []
    for aa in all_contacts:
        mapped = mapping[aa]
        lddts.append(aa_to_lddt[mapped])
    return lddt_list[torch.tensor(lddts)]
