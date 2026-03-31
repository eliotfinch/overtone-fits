#!/home/eliot.finch/ringdown/SURF24/.venv/bin/python

import numpy as np
import pandas as pd

import qnmfits
import utils

ID = 1
sim_info = utils.load_cce_data(ID)
sim = sim_info['sim']
chif = sim.chif_mag
Mf = sim.Mf

t0_E = pd.read_csv('../data/t0N_E.csv', index_col=0)
t0 = t0_E[str(ID)].values[10]

N = 13
modes = [(2, 2, n, 1) for n in range(N+1)] + [(3, 2, 0, 1)]

delta_list_dense = np.linspace(-0.2, 0.2, 100)

for ntilde in range(N+1):

    print(f'Calculating epsilon grid for ntilde={ntilde}...')

    epsilon_grid = np.zeros((len(delta_list_dense), len(delta_list_dense)))

    for i, delta_i in enumerate(delta_list_dense[::-1]):
        for j, delta_r in enumerate(delta_list_dense):
            frequency_deltas_r = np.zeros(len(modes))
            frequency_deltas_i = np.zeros(len(modes))
            frequency_deltas_r[ntilde] = delta_r
            frequency_deltas_i[ntilde] = delta_i
            epsilon, Mf_bestfit, chif_bestfit = qnmfits.calculate_epsilon(
                times=sim.times,
                data=sim.h[2, 2],
                modes=modes,
                Mf=sim.Mf,
                chif=sim.chif_mag,
                t0=t0,
                t0_method='closest',
                T=100,
                delta_r=frequency_deltas_r,
                delta_i=frequency_deltas_i
            )
            epsilon_grid[i, j] = epsilon

    np.save(f'epsilon_grid_n{ntilde}.npy', epsilon_grid)
