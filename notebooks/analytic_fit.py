import numpy as np
import pandas as pd

from scipy.special import erfc
from scipy.linalg import cholesky, cho_solve, solve
from scipy.stats import gaussian_kde

rng = np.random.default_rng()


def calculate_weights(A, A_scale):
    """
    Compute the weights required to transform to a prior uniform in the
    magnitude of the amplitude (A) for a fixed Gaussian amplitude prior on
    Re[C] and Im[C].

    Parameters
    ----------
    A : array_like
        The (unweighted) posterior on amplitude, A
    A_scale : float
        The standard deviation of the Gaussian amplitude prior.

    Returns
    -------
    weights : np.ndarray
        Importance weights for each A value.
    """
    return (A_scale**2/A)*np.exp(A**2/(2*A_scale**2))


def calculate_marg_weights(A, A_scale_max, A_max):
    """
    Compute the weights required to transform to a prior uniform in the
    magnitude of the amplitude (A) whilst marginalizing over the scale of the
    Gaussian amplitude prior Re[C] and Im[C].

    Parameters
    ----------
    A : array_like
        The (unweighted) posterior on amplitude, A
    a_scale_max : float
        The maximum value of the uniform prior on a_scale.
    a_max : float
        The maximum radius for the uniform target prior on a.

    Returns
    -------
    weights : np.ndarray
        Importance weights for each A value.
    """
    arg = A/(np.sqrt(2)*A_scale_max)
    erfc_vals = erfc(arg)

    weights = (A_scale_max*np.sqrt(2)/(A_max*np.sqrt(np.pi)))/erfc_vals

    return weights


def sample(times, data, sigma, qnm_omegas, t0, t0_method='geq', T=100,
           t_ref=None, A_scales=None, marginalize_A_scale=False, reweight=True,
           n_samples=50000):
    """
    times (N), parameters (K)

    Hogg et al. http://arxiv.org/abs/2005.14199

    Parameters
    ----------
    times : array_like

    data : dict
        Keys 'plus' and 'cross'.

    sigma : float

    qnms_omegas : list of complex frequencies
        Length K/2

    t0 : float

    t0_method : str, optional [Default 'geq']

    T : float, optional [Default: 100]

    t_ref : float, optional [Default: None]
        The time at which the complex amplitudes are defined. If None,
        t_ref = t0.

    A_scales : array_like, optional [Default: None]
        Length K/2. The standard deviation of the Gaussian amplitude prior.
        This can be different for each QNM If None, the limit
        Lambda -> infinity is used (a flat prior on Re[C] and Im[C]).

    marginalize_A_scale : bool, optional [Default: False]
        If True, marginalize over A_scale with a uniform prior from zero to
        A_scale.

    reweight : bool, optional [Default: True]

    n_samples : float, optional [Default: 50000]
    """

    # Mask the data with the requested method
    if t0_method == 'geq':

        data_mask = (times >= t0) & (times < t0+T)

        times = times[data_mask]

        h_dict = {}
        for pol, h in data.items():
            h_dict[pol] = h[data_mask]

        sigma = sigma[data_mask]

    elif t0_method == 'closest':

        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)

        times = times[start_index:end_index]

        h_dict = {}
        for pol, h in data.items():
            h_dict[pol] = h[start_index:end_index]

        sigma = sigma[start_index:end_index]

    else:
        raise ValueError(
            "Requested t0_method is not valid. Please choose between 'geq'and "
            "'closest'"
        )

    # N = len(times)
    K = 2*len(qnm_omegas)

    if t_ref is None:
        t_ref = t0

    dt = t_ref - t0

    # Construct the design matrices, shape (N,K)

    M_plus = []
    M_cross = []

    for omega_complex in qnm_omegas:

        tau = -1/np.imag(omega_complex)
        omega = np.real(omega_complex)

        # Evaluate the cosine and sine terms
        cos_term = np.exp(-(times-t0)/tau)*np.cos(omega*(times-t0))
        sin_term = np.exp(-(times-t0)/tau)*np.sin(omega*(times-t0))

        # Add to the design matrices
        M_plus.append(cos_term)
        M_plus.append(sin_term)
        M_cross.append(sin_term)
        M_cross.append(-cos_term)

    # Ensure M is shape (N,K)
    M_plus = np.array(M_plus).T
    M_cross = np.array(M_cross).T

    M_dict = {'plus': M_plus, 'cross': M_cross}

    # If no A_scale is specified, we take the limit Lambda -> infinity
    if A_scales is None:

        # Initialise
        A_inv = np.zeros((K, K))
        x = np.zeros(K)

        # Iterate
        for pol in ['plus', 'cross']:

            h = h_dict[pol]
            M = M_dict[pol]

            # MT_Cinv_M = np.dot(M.T, M/sigma**2)
            MT_Cinv_M = np.dot(M.T, M/(sigma**2)[:, np.newaxis])
            A_inv += MT_Cinv_M

            # MT_Cinv_y = np.dot(M.T, h)/sigma**2
            MT_Cinv_y = np.dot(M.T, h/sigma**2)
            x += MT_Cinv_y

        # A = np.linalg.inv(A_inv)
        # a = np.dot(A, x)

        # Draw samples. The samples array has shape (n_samples, K), where
        # columns are ordered Re[C_0], Im[C_0], Re[C_1], Im[C_1], ...
        # samples = rng.multivariate_normal(a, A, size=n_samples)

        # Or if I want to avoid inverting...
        A_inv_chol = cholesky(A_inv, lower=True)
        a = cho_solve((A_inv_chol, True), x)

        samples = np.zeros((n_samples, K))
        for i in range(n_samples):
            z = rng.standard_normal(K)
            samples[i] = a + solve(A_inv_chol.T, z)

    else:

        # If marginalize_A_scale is False, A_scale if fixed to whatever is
        # provided and we just have one Lambda
        if not marginalize_A_scale:

            # Construct Lambda inverse
            Lambda_diag = np.zeros(K)
            for n in range(K//2):
                Lambda_diag[2*n:2*n+2] = A_scales[n]**2
            Lambda_inv = np.diag(1/Lambda_diag)

            # Initialise
            A_inv = Lambda_inv.copy()
            x = np.zeros(K)

            # Iterate
            for pol in ['plus', 'cross']:

                h = h_dict[pol]
                M = M_dict[pol]

                MT_Cinv_M = np.dot(M.T, M)/sigma**2
                A_inv += MT_Cinv_M

                MT_Cinv_y = np.dot(M.T, h)/sigma**2
                x += MT_Cinv_y

            # A = np.linalg.inv(A_inv)
            # a = np.dot(A, x)

            # samples = rng.multivariate_normal(a, A, size=n_samples)

            # Or if I want to avoid inverting...
            A_inv_chol = cholesky(A_inv, lower=True)
            a = cho_solve((A_inv_chol, True), x)

            samples = np.zeros((n_samples, K))
            for i in range(n_samples):
                z = rng.standard_normal(K)
                samples[i] = a + solve(A_inv_chol.T, z)

        # Otherwise, marginalize over A_scale. Note that for this to work we
        # need to actually compute the likelihood and do sampling. I think this
        # would require computing b and B (Eqs. 14 and 15 in Hogg et al.
        # http://arxiv.org/abs/2005.14199)
        else:

            A_scale_array = np.zeros((n_samples, K//2))
            for i, A_scale_max in enumerate(A_scales):
                A_scale_array[:, i] = rng.uniform(0, A_scale_max, n_samples)

            # Initialise samples array
            samples = np.zeros((n_samples, K))

            for i, A_scale in enumerate(A_scale_array):

                # Construct Lambda inverse
                Lambda_diag = np.zeros(K)
                for n in range(K//2):
                    Lambda_diag[2*n:2*n+2] = A_scale[n]**2
                Lambda_inv = np.diag(1/Lambda_diag)

                # Initialise
                A_inv = Lambda_inv.copy()
                x = np.zeros(K)

                # Iterate
                for pol in ['plus', 'cross']:

                    h = h_dict[pol]
                    M = M_dict[pol]

                    MT_Cinv_M = np.dot(M.T, M)/sigma**2
                    A_inv += MT_Cinv_M

                    MT_Cinv_y = np.dot(M.T, h)/sigma**2
                    x += MT_Cinv_y

                # Code which actually inverts A_inv
                A = np.linalg.inv(A_inv)
                a = np.dot(A, x)

                samples[i] = rng.multivariate_normal(a, A, size=1)

                # Or if I want to avoid inverting... (this doesn't work)
                # A_inv_chol = cholesky(A_inv, lower=True)
                # a = cho_solve((A_inv_chol, True), x)

                # z = rng.multivariate_normal(np.zeros(K), np.eye(K), size=1).T
                # samples[i] = (
                #     a.reshape((K, 1)) + solve(A_inv_chol, z)
                # ).flatten()

    # Propagate the complex amplitudes to t_ref, convert to amplitude and
    # phase, and reweight to have a flat prior on amplitude and phase

    A_samples = []
    phi_samples = []

    for i, omega in enumerate(qnm_omegas):

        # Construct the complex amplitude for this QNM
        Re_Ci = samples[:, 2*i]
        Im_Ci = samples[:, 2*i+1]
        Ci = Re_Ci + 1j*Im_Ci

        # Propagate the complex amplitude to t_ref
        Ci *= np.exp(-1j*omega*dt)

        # Get amplitude and phase
        Ai = np.abs(Ci)
        phi_i = np.angle(Ci)

        A_samples.append(Ai)
        phi_samples.append(phi_i)

    # Ensure that A_samples and phi_samples have shape (n_samples, K/2)
    A_samples = np.array(A_samples).T
    phi_samples = np.array(phi_samples).T

    # Reweight the samples to have a flat prior on the amplitude magnitude
    if reweight:

        if A_scales is None:

            # For the limit Lambda -> infinity, the effective prior on A is
            # p(A) ~ A. So, divide through by this to reweight.
            weights = np.ones(n_samples)
            for i, A_sample in enumerate(A_samples):
                weights[i] = np.prod(1/A_sample)

        else:

            if not marginalize_A_scale:

                A_maxs = A_scales

                individual_mode_weights = [
                    calculate_weights(A_samples[:, n], A_scales[n])
                    for n in range(K//2)
                ]
                weights = np.prod(individual_mode_weights, axis=0)

            else:

                # We could have a different upper bound on the uniform prior on
                # A, but for now we set it to A_scales
                A_maxs = A_scales

                # For finite Lambda, the effective prior is a bit more
                # complicated.
                individual_mode_weights = [
                    calculate_marg_weights(
                        A_samples[:, n], A_scales[n], A_maxs[n]
                    )
                    for n in range(K//2)
                ]
                weights = np.prod(individual_mode_weights, axis=0)

        # Normalize the weights
        weights /= np.sum(weights)

        neff = sum(weights)**2/sum(weights**2)
        print(f"Effective number of samples: {neff}")

        # Resample
        reweight_indices = rng.choice(n_samples, size=n_samples, p=weights)
        A_samples = A_samples[reweight_indices]
        phi_samples = phi_samples[reweight_indices]

    return samples, A_samples, phi_samples


def unwrap(a, period: float = 2*np.pi):
    """
    Takes in an array of phases which wraps around at some period, and returns
    the array, transformed so that the distribution has no discontinuity.
    """

    # Sort the array but keep the indices to unsort it
    sort_indices = np.argsort(a)
    unsort_indices = np.argsort(sort_indices)
    a_sorted = a[sort_indices]

    # Find the best point to split at, determined by the lowest standard
    # deviation
    cut_std = []
    for i in range(len(a_sorted)):
        cut_std.append(
            np.std(np.concatenate([a_sorted[:i], a_sorted[i:]-period]))
        )
    index = np.argmin(cut_std)

    # Perform the split and unsort
    unwrapped_array = np.concatenate(
        [a_sorted[:index], a_sorted[index:]-period]
    )[unsort_indices]

    return unwrapped_array


# functions to build KDEs and calculate credible intervals

def build_kde(samples):
    """
    Builds KDE with no mirroring.
    """
    return gaussian_kde(samples)


def calculate_interval(pdf, x, credible_interval=0.9):
    """
    Compute the highest density interval for a given PDF evaluated at x.

    Imagine moving a horizontal line from the top to the bottom of the PDF. We
    compute the area under the PDF between the two points where the line
    intersects the PDF. When this area is equal to the credible interval, we
    have found the highest density interval.
    """
    # Sort the pdf and x values from the highest to the lowest pdf value
    sorted_indices = np.argsort(pdf)[::-1]
    x_sorted = x[sorted_indices]
    pdf_sorted = pdf[sorted_indices]

    # Compute the area between the horizontal line and the PDF for each line
    # height
    cumulative = np.cumsum(pdf_sorted)
    cumulative /= cumulative[-1]

    # Find the smallest interval containing the specified credible mass
    interval_idx = np.where(cumulative >= credible_interval)[0][0]
    hdi = (
        x_sorted[:interval_idx + 1].min(), x_sorted[:interval_idx + 1].max()
    )

    return hdi


def A_kde(samples):
    """
    Build a kde for amplitudes.Returns the pdf, and the points it is evaluated
    on.
    """

    bounds = [0, max(samples)]

    # List of values to evaluate the KDE over
    x = np.linspace(*bounds, 1000)

    if np.median(samples) < 3*np.std(samples):
        samples_mirror = np.concatenate([-samples, samples])
        kde_built = build_kde(samples_mirror)
        return 2*kde_built(x), x
    else:
        kde_built = build_kde(samples)
        return kde_built(x), x


def phi_kde(samples):
    """
    Build a kde for phases. Returns the pdf, and the points it is evaluated on.
    """
    samples_unwrapped = unwrap(samples)
    kde_built = build_kde(samples_unwrapped)
    bounds = [min(samples_unwrapped), max(samples_unwrapped)]
    x = np.linspace(*bounds, 1000)
    return kde_built(x), x


def overlap(interval1: tuple, interval2: tuple, is_phase=False) -> bool:
    """
    Takes in two intervals as tuples and returns whether they overlap or not.
    """

    # use the pandas interval objects to do this
    i1_pd = pd.Interval(interval1[0], interval1[1], closed='both')
    i2_pd = pd.Interval(interval2[0], interval2[1], closed='both')

    if i1_pd.length == 0 or i2_pd.length == 0:
        # Want it to return False if given an interval of zero width
        return False
    else:
        return i1_pd.overlaps(i2_pd)


def resolved(interval_time_series) -> int:
    """Takes a time series of (amplitude) intervals, and determines the index
    of the last interval which is not consistent with 0. If no interval is
    consistent with 0, then the function returns the index of the last
    interval."""

    for i, interval in enumerate(interval_time_series):
        if interval[0] <= 0:
            return i - 1

    return i


def stable(interval_time_series, last_resolved: int, is_phase=False) -> int:
    """Takes a time series of intervals, and determines the index of the first
    stable time."""

    # We start at the last resolved interval, and work backwards until we
    # encounter an interval which isn't consistent with all intervals between
    # that interval and the last resolved interval
    stable_intervals = [interval_time_series[last_resolved]]

    for i, interval in reversed(
        list(enumerate(interval_time_series[:last_resolved]))
    ):

        # For amplitudes we don't need to worry about periodicity
        if not is_phase:

            # Test the overlap between this interval and all stable intervals
            # so far
            overlaps = [
                overlap(interval, stable_interval, is_phase)
                for stable_interval in stable_intervals
            ]

            # If the current interval does not overlap with a later interval,
            # then we have found the latest unstable time (so the beginning of
            # the stable window is at i+1)
            if False in overlaps:
                return i+1
            else:
                stable_intervals.append(interval)

        # For phases we need to consider shifts by 2pi when testing the
        # overlap. Note that we might want to take the interval which has the
        # most overlap with the stable intervals.
        if is_phase:

            # Unshifted interval
            overlaps_unshifted = [
                overlap(interval, stable_interval, is_phase)
                for stable_interval in stable_intervals
            ]

            # If the unshifted interval is consistent with all the stable
            # intervals, then append it to the list of stable intervals and
            # move onto the next interval
            if False not in overlaps_unshifted:
                stable_intervals.append(interval)
                continue

            # Interval shifted up by 2pi
            interval_shifted_up = (
                interval[0] + 2*np.pi, interval[1] + 2*np.pi
            )
            overlaps_shifted_up = [
                overlap(interval_shifted_up, stable_interval, is_phase)
                for stable_interval in stable_intervals
            ]

            # Repeat the overlap check
            if False not in overlaps_shifted_up:
                stable_intervals.append(interval_shifted_up)
                continue

            # Interval shifted down by 2pi
            interval_shifted_down = (
                interval[0] - 2*np.pi, interval[1] - 2*np.pi
            )
            overlaps_shifted_down = [
                overlap(interval_shifted_down, stable_interval, is_phase)
                for stable_interval in stable_intervals
            ]

            # If False is in this list, then the interval does not overlap with
            # any of the stable intervals, so we have found the latest unstable
            # time
            if False in overlaps_shifted_down:
                return i+1
            else:
                stable_intervals.append(interval_shifted_down)

    # If we get through all the the intervals, then 0 is the first stable time
    return 0
