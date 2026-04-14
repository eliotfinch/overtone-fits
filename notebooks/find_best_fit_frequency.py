#!/home/eliot.finch/ringdown/SURF24/env/bin/python

import numpy as np
import pandas as pd

import qnmfits
import utils

ID = 1
sim_info = utils.load_cce_data(ID)
sim = sim_info['sim']
chif = sim.chif_mag
Mf = sim.Mf

N_max = 15
mode_list = utils.get_mode_list(N_max)

t0_E = pd.read_csv('../data/t0N_E.csv', index_col=0)

for k, t0 in enumerate(t0_E[str(ID)].values[:11]):

    if k == 0:
        continue

    print(f"t0_{k} = {t0}", flush=True)

    epsilon = 1

    for j, modes in enumerate(mode_list):

        new_epsilon = qnmfits.calculate_epsilon(
            sim.times,
            sim.h[2, 2],
            modes + [(3, 2, 0, 1)],
            Mf,
            chif,
            t0=t0,
            t0_method='closest',
        )[0]

        if new_epsilon < epsilon:
            epsilon = new_epsilon
        else:
            break

    # This is the number of modes suitable for this start time (that is, not
    # late to be over fitting, but not so early that the fit is bad).
    N = j - 1
    print(f"N = {N}, epsilon = {epsilon}", flush=True)

    modes = [(2, 2, n, 1) for n in range(N+1)] + [(3, 2, 0, 1)]

    omega_bf_list = []

    for ntilde in range(N+1):

        print(
            f'Calculating best-fit frequency for ntilde={ntilde}...',
            flush=True
        )

        free_modes = [(2, 2, ntilde, 1)]
        fixed_modes = [mode for mode in modes if mode not in free_modes]

        omega_bf_list.append(qnmfits.free_frequency_fit(
            sim.times,
            sim.h[2, 2],
            t0=t0,
            t0_method='closest',
            modes=fixed_modes,
            Mf=Mf,
            chif=chif
        ))

    np.save(
        f'../data/best_fit_frequencies/t0_{k}_N{N}.npy',
        omega_bf_list
    )
