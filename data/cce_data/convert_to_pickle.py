import os
import shutil
import pickle
import scri

from pathlib import Path

# My qnmfits package takes a dictionary of waveform modes. Convert each CCE
# h5 file to a dictionary and save via pickle.

cce_dir = Path('/Users/eliot/Documents/Research/Ringdown/sxs-collaboration/qnmfits/qnmfits/data')
level = 4

for ID in range(1, 14):

    # Destination directory
    save_dir = Path(f'SXS_BBH_ExtCCE_{ID:04d}/Lev{level}')
    save_dir.mkdir(exist_ok=True, parents=True)

    # Simulation directory
    sim_dir = cce_dir / f'SXS:BBH_ExtCCE:{ID:04d}/Lev{level}'
    
    # List of files in the simulation directory
    file_list = os.listdir(sim_dir)

    for filename in file_list:

        if filename.endswith('superrest.h5'):

            h = scri.SpEC.file_io.read_from_h5((sim_dir / filename).as_posix())

            # Convert WaveformModes to a dictionary of modes
            h_dict = {'times': h.t}
            for ell in range(h.ell_min, h.ell_max + 1):
                for m in range(-ell, ell + 1):
                    h_dict[ell, m] = h.data[:,h.index(ell,m)]
            
            # Save h_dict as a pickle
            new_filename = f"{filename.split('.')[0]}.pickle"
            with open(save_dir / new_filename, 'wb') as f:
                pickle.dump(h_dict, f)

        if filename.endswith('superrest.json'):

            # Copy metadata
            shutil.copy(sim_dir / filename, save_dir / filename)
        