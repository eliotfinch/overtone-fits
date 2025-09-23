#!/home/eliot.finch/ringdown/SURF24/.venv/bin/python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import qnmfits
import analytic_fit
import utils
import corner
import emcee

from pathlib import Path
from multiprocessing import Pool

plt.rcParams.update(utils.rcparams)

rng = np.random.default_rng()

ID = 4
N = 11
T = 100

data_dir = Path('../data/emcee')
run_dir = data_dir / f'ID{ID}_N{N}_t0M_T{T}'
run_dir.mkdir(parents=True, exist_ok=True)

sim_info = utils.load_cce_data(ID)
sim = sim_info['sim']
chif = sim.chif_mag
Mf = sim.Mf

t0_M = pd.read_csv('../data/t0_data_30_to_100.csv', index_col=0)
t0 = t0_M[str(ID)].values[N]

modes = [(2, 2, n, 1) for n in range(N+1)]
modes_with_320 = [(2, 2, n, 1) for n in range(N+1)] + [(3, 2, 0, 1)]

best_fit_with_320 = qnmfits.ringdown_fit(
    sim.times,
    sim.h[2, 2],
    modes=modes_with_320,
    Mf=Mf,
    chif=chif,
    t0=t0,
    t0_method='closest'
)

A320 = np.abs(best_fit_with_320['C'])[-1]

ls_fit = qnmfits.ringdown_fit(
    sim.times,
    sim.h[2, 2],
    modes=modes,
    Mf=Mf,
    chif=chif,
    t0=t0,
    t0_method='closest'
)

omega = ls_fit['frequencies']
ls_re_c = np.real(ls_fit['C'])
ls_im_c = np.imag(ls_fit['C'])

ls_re_c_im_c = []
for re_c, im_c in zip(ls_re_c, ls_im_c):
    ls_re_c_im_c.append(re_c)
    ls_re_c_im_c.append(im_c)

ls_a = np.abs(ls_fit['C'])
ls_phi = np.angle(ls_fit['C'])
ls_a_phi = np.concatenate((ls_a, ls_phi))

data = {
    'plus': np.real(sim.h[2, 2]),
    'cross': -np.imag(sim.h[2, 2]),
}

samples, A_samples, phi_samples = analytic_fit.sample(
    times=sim.times,
    data=data,
    sigma=A320*np.ones_like(sim.times),
    qnm_omegas=omega,
    t0=t0,
    t0_method='closest',
    T=T,
    reweight=False,
)

start_index = np.argmin((sim.times-t0)**2)
end_index = np.argmin((sim.times-t0-T)**2)

analysis_times = sim.times[start_index:end_index] - t0

analysis_data = {}
for pol, h in data.items():
    analysis_data[pol] = h[start_index:end_index]

sigma = A320*np.ones_like(analysis_times)

lower = 2*np.min(samples)
upper = 2*np.max(samples)
stds = np.std(samples, axis=0)


def ringdown_model(theta, times, frequencies):
    """
    Ringdown model class.

    Parameters
    ----------
    times : array-like
        The time values at which the model is evaluated.
    theta : dict
        Contains re[C] and im[C] for each requested mode.
    frequencies : array-like
        The frequencies corresponding to each qnm.
    """

    model = np.zeros_like(times, dtype=complex)

    real_amplitudes = theta[::2]
    imag_amplitudes = theta[1::2]

    for re_c, im_c, omega in zip(
        real_amplitudes, imag_amplitudes, frequencies
    ):
        model += (re_c+1j*im_c)*np.exp(-1j*omega*times)

    model_dict = {'plus': np.real(model), 'cross': -np.imag(model)}

    return model_dict


def log_likelihood(theta, data, times, sigma, frequencies):

    h_theta = ringdown_model(theta, times, frequencies)

    log_Lplus = -0.5*sum(((data['plus']-h_theta['plus'])/sigma)**2)
    log_Lcross = -0.5*sum(((data['cross']-h_theta['cross'])/sigma)**2)

    return log_Lplus + log_Lcross


def log_prior(theta, lower, upper):
    # Check whether any param is outside of its appropriate range
    if np.all(np.logical_and(lower <= theta, theta <= upper)):
        return 0.0
    return -np.inf


def log_probability(theta, data, times, sigma, frequencies, lower, upper):

    # Calculate log prior. If log prior = 0, then probability is 0.
    logp = log_prior(theta, lower, upper)
    if not np.isfinite(logp):
        return -np.inf

    # Return log prior + log likelihood
    return logp + log_likelihood(theta, data, times, sigma, frequencies)


# Pick a set of random starting locations around the least-squares result
pos = ls_re_c_im_c + rng.standard_normal((64, 2*len(modes)))
nwalkers, ndim = pos.shape

with Pool(processes=16) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        pool=pool,
        args=(
            analysis_data,
            analysis_times,
            sigma,
            omega,
            lower,
            upper
        ),
    )
    sampler.run_mcmc(pos, 100000, progress=True)

tau = sampler.get_autocorr_time(quiet=True)

emcee_samples = sampler.get_chain(
    discard=int(2*np.mean(tau)),
    thin=int(np.mean(tau)/2),
    flat=True
    )

real_labels = [
    rf'$\mathrm{{Re}}[C_{{{ell}{m}{n}}}]$' for (ell, m, n, _) in modes
]
imag_labels = [
    rf'$\mathrm{{Im}}[C_{{{ell}{m}{n}}}]$' for (ell, m, n, _) in modes
]
labels = []
for real, imag in zip(real_labels, imag_labels):
    labels.append(real)
    labels.append(imag)

fig = corner.corner(
    samples,
    levels=[0.9],
    labels=labels,
    data_kwargs={
        'alpha': 0.005,
    },
    hist_kwargs={
        'density': True,
    }
)
corner.corner(
    emcee_samples,
    levels=[0.9],
    no_fill_contours=True,
    plot_datapoints=False,
    color='C3',
    hist_kwargs={
        'density': True,
    },
    fig=fig
)
fig.savefig(run_dir / 'corner_comparison.png', dpi=180)

np.savetxt(run_dir / 'samples.dat', samples)
np.savetxt(run_dir / 'emcee_samples.dat', emcee_samples)
