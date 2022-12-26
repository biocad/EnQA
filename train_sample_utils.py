import os
from pathlib import Path
import random
import pandas as pd
from re import compile

def get_ids_for_benchmark(path_to_structures: Path):
    complex_dirs = os.listdir(path_to_structures)
    res = set()
    for comp_dir in complex_dirs:
        i = 1
        if "constr" in comp_dir:
            i = 2
        splitted = comp_dir.split("_")
        pdb_id = splitted[i]
        res.add(pdb_id)
    return res


def create_docked_equals_real(path_to_structures:Path,prefix:str='real',files_real=['real.pdb','joined_real.pdb'],files_docked=['docked.pdb','joined_docked.pdb'],file_lddt='lddt.csv'):
    path_to_structures=Path(path_to_structures)
    suffix='_'.join(path_to_structures.stem.split('_')[-2:])
    dir_name=path_to_structures.parent/'_'.join([prefix,suffix])
    dir_name.mkdir(exist_ok=True)
    for source,dest in zip(files_real,files_docked):
        p_source=path_to_structures/source
        p_dest_source=dir_name/source
        p_dest=dir_name/dest
        p_dest_source.write_text(p_source.read_text())
        p_dest.write_text(p_source.read_text())
    lddt=pd.read_csv(path_to_structures/file_lddt, sep='\t', skiprows=10)
    lddt['Score']=1
    lddt.to_csv(dir_name/file_lddt)
    return dir_name.name

def get_sample(path_to_structures: Path,size_ids=100,size_constraits=5,size_non_constraits=5,size_real=1,prefix_real='real',seed =42):
    random.seed(seed)
    complex_dirs = os.listdir(path_to_structures)
    pdb_ids=get_ids_for_benchmark(path_to_structures)
    sample_ids=random.sample(pdb_ids,size_ids)
    sample_list=[]
    for id in sample_ids:
        sample_list_id=[]
        if size_constraits>0:
            r=compile(fr'\w*_constr_{id}\w*')
            constraits_dirs=list(filter(r.match, complex_dirs))
            sample_list_id+=random.sample(constraits_dirs,size_constraits)
        if size_non_constraits>0:
            r=compile(fr'\d*_{id}_\w*')
            non_constraits_dirs=list(filter(r.match,complex_dirs))
            sample_list_id+=random.sample(non_constraits_dirs,size_non_constraits)
        a=create_docked_equals_real(path_to_structures/sample_list_id[0],prefix=prefix_real)
        sample_list_id+=size_real*[a]
        sample_list+=sample_list_id
    return(sample_list)
    
if __name__=='__main__':
    a=get_sample(Path('/mnt/volume_complex_lddt/consistent_alpha_hedge'))
    print(*a,sep='\n')
