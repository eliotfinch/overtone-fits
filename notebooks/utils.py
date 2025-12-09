import numpy as np

import json
import pickle
import qnmfits
import math

from scipy.signal import find_peaks
from pathlib import Path
from matplotlib import text as mtext

# This is a useful package for finding the knee of a curve. We will use it to
# determine when a ringdown model becomes a good fit. See
# https://pypi.org/project/kneed/
from kneed import KneeLocator

rcparams = {
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif',
    'font.sans-serif': ['Computer Modern Roman'],
    'text.usetex': True
}


class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve. Taken from
    https://stackoverflow.com/a/44521963
    """
    def __init__(self, x, y, text, axes, **kwargs):

        super(CurvedText, self).__init__(x[0], y[0], ' ', **kwargs)

        axes.add_artist(self)

        # Saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        # Creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                # Make this an invisible 'a':
                t = mtext.Text(0, 0, 'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0, 0, c, **kwargs)

            # Resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder + 1)

            self.__Characters.append((c, t))
            axes.add_artist(t)

    # Overloading some member functions, to assure correct functionality
    # on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c, t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self, renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        # Preparations

        # Determining the aspect ratio:
        # from https://stackoverflow.com/a/42014041/2454357

        # Data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        # Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        # Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        # Final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        # Points of the curve in figure coordinates:
        x_fig, y_fig = (
            np.array(t) for t in zip(*self.axes.transData.transform(
                [(i, j) for i, j in zip(self.__x, self.__y)])
            )
        )

        # Point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        # Arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist), 0, 0)

        # Angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]), (x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)

        rel_pos = 10
        for c, t in self.__Characters:
            # Finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1 = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            # Ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            # Finding the two data points between which the horizontal center
            # point of the character will be situated left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            # If we exactly hit a data point:
            if ir == il:
                ir += 1

            # How much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            # Relative distance between il and ir where the center of the
            # character will be
            fraction = (w/2-used)/r_fig_dist[il]

            # Setting the character position in data coordinates:
            # interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            # Getting the offset when setting correct vertical alignment in
            # data coordinates
            t.set_va(self.get_va())
            bbox2 = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            # The rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            # Computing the offset vector of the rotated character
            drp = np.dot(dr, rot_mat)

            # Setting final position and rotation:
            t.set_position(np.array([x, y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            # Updating rel_pos to right edge of character
            rel_pos += w-used


file_path = Path(__file__).parent
cce_dir = file_path / '../data/cce_data'

# For convenience there is a file in the cce_data directory that contains
# useful information about each of the simulations
with open(cce_dir / 'cce-catalog.json', 'r') as f:
    cce_catalog = json.load(f)


def load_cce_data(
        ID: int,
        cce_catalog: list = cce_catalog,
        cce_dir: str = cce_dir,
        lev: int = 5
):
    """
    Load the CCE data for a given simulation ID.

    Parameters
    ----------
    ID: int
        The ID number of the simulation
    cce_catalog : list (optional)
        List of dictionaries with properties of each CCE sim.
    cce_dir : str (optional)
        The directory in which the CCE data can be found.
    lev : int (optional)
        The resolution level of the simulation to be used. Default is 5.

    Returns
    -------
    sim_info : dictionary
        Keys:
            'name' : str
                The simulation name.
            'preferred_R' : float
                The preferred radius for the simulation.
            'url' : str
                Link to the simulation on zenodo.org
            'h' : ndarray
                A time series of the strain.
            'times' : ndarray
                Array of the times the simulation took place over.
            'metadata' : dict
                Provides the metadata for the remnant
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
    """
    # Look up the appropriate simulation in the catalog
    for entry in cce_catalog:
        if entry['name'] == f'SXS_BBH_ExtCCE_{ID:04d}':
            sim_info = entry

    # Check that the simulation was found
    assert 'sim_info' in locals(), "Simulation not found in directory given."

    # Load the data for this simulation
    data_filename = (
        f"rhOverM_BondiCce_R{int(sim_info['preferred_R']):04d}"
        "_superrest.pickle"
    )
    with open(
        cce_dir / f"{sim_info['name']}" / f'Lev{lev}' / data_filename, 'rb'
    ) as f:
        h = pickle.load(f)

    # Load the remnant properties
    metadata_filename = (
        f"metadata_BondiCce_R{int(sim_info['preferred_R']):04d}"
        "_superrest.json"
    )
    with open(
        cce_dir / f"{sim_info['name']}" / f'Lev{lev}' / metadata_filename, 'r'
    ) as f:
        metadata = json.load(f)

    # We want the timestamps as a separate array. This also removes it from
    # the h dictionary
    times = h.pop('times')

    # Create a sim in qnmfits for the data
    sim = qnmfits.Custom(
        times,
        h,
        metadata,
        zero_time=(2, 2)
    )

    sim_info['metadata'] = metadata
    sim_info['sim'] = sim

    return sim_info


def get_mode_list(Nmax: int, additional_modes: list = []):
    """
    Get a list of lists of the first k overtones of the 220 mode for
    0<=k<=Nmax. Adds any additional modes if desired.
    """
    return [
        [(2, 2, n, 1) for n in range(N)] + additional_modes
        for N in range(1, Nmax+2)
    ]


def generate_mismatch_curve(
        ID: int,
        Nmax: int,
        cce_catalog: list = cce_catalog,
        cce_dir: str = cce_dir,
        additional_modes=[]
):
    """
    Generate the mismatch curves for each N up to Nmax for a given simulation.

    Parameters
    ----------
    ID : int OR dict
        The ID number of the simulation or the simulation info as a dictionary
    Nmax : int
        The maximum number of overtones to be fitted.
    cce_catalog : list (optional)
        List of dictionaries with properties of each CCE sim.
    cce_dir : str (optional)
        The directory in which the CCE data can be found.
    additional_modes : list (optional)
        Any additional modes which should be added to the simulation. Default
        is for none to be added.

    Returns
    -------
    sim_info : dictionary
        Keys:
            'name' : str
                The simulation name.
            'preferred_R' : float
                The preferred radius for the simulation.
            'url' : str
                Link to the simulation on zenodo.org
            'h' : ndarray
                A time series of the strain.
            'times' : ndarray
                Array of the times the simulation took place over.
            'metadata' : dict
                Provides the metadata for the remnant
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
            'mm_list' : ndarray
                Array of the mismatches at each timestep.
    """
    if type(ID) is int:
        sim_info = load_cce_data(ID, cce_catalog, cce_dir)
    else:
        assert type(ID) is dict, (
            "Check the stored value of ID, or input ID as an int to load data."
        )
        sim_info = ID
    sim = sim_info['sim']
    t0_array = sim.times[(sim.times > -30) & (sim.times < 100)]

    # Check whether the mismatch has already been calculated (save time!)
    if (
        'mm_lists' not in sim_info.keys() or
        ('mm_lists' in sim_info.keys() and len(sim_info['mm_lists']) < Nmax)
    ):
        sim_info['mm_lists'] = []
        mode_list = get_mode_list(Nmax, additional_modes)
        for i, modes in enumerate(mode_list):
            # Create mismatch curve. There is a function in qnmfits that
            # automatically performs a ringdown fit for an array of start
            # times, and returns a mismatch list.
            print(f'\rCalculating for n = {i}')
            mm_list = qnmfits.mismatch_t0_array(
                sim.times,
                sim.h[2, 2],
                modes,
                Mf=sim.Mf,
                chif=sim.chif_mag,
                t0_array=t0_array,
                t0_method='closest'
            )
            sim_info['mm_lists'].append(mm_list)
    else:
        print("Mismatch lists already calculated.")

    return sim_info


def t0NM_finder(
        ID,
        Nmax: int,
        cce_catalog: list = cce_catalog,
        cce_dir: str = cce_dir
):
    """
    Finds the locations of t0N for a simulation, based on the "knee" of the
    mismatch curve.

    Parameters
    ----------
    ID : int or dict
        The ID number of the simulation or the simulation info as a dictionary
    Nmax : int
        Number of overtones you want t0 for
    cce_catalog : list (optional)
        List of dictionaries with properties of each CCE sim.
    cce_dir : str (optional)
        The directory in which the CCE data can be found.

    Returns
    -------
    sim_info : dictionary
        Keys:
            'name' : str
                The simulation name.
            'preferred_R' : float
                The preferred radius for the simulation.
            'url' : str
                Link to the simulation on zenodo.org
            'h' : ndarray
                A time series of the strain.
            'times' : ndarray
                Array of the times the simulation took place over.
            'metadata' : dict
                Provides the metadata for the remnant
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
            'mm_list' : ndarray
                Array of the mismatches at each timestep.
            't0M_list' : list
                List of [t0N, MM(t0N)] for each N.
    """
    # Check whether the mismatch curve has already been calculated, calculates
    # it otherwise
    if (
        type(ID) is dict
        and 't0M_list' in ID.keys()
        and len(ID['t0M_list']) >= Nmax
    ):
        sim_info = ID
        return sim_info
    if not (
        type(ID) is dict
        and 'mm_lists' in ID.keys()
        and len(ID['mm_lists']) >= Nmax
    ):
        sim_info = generate_mismatch_curve(
            ID, Nmax, cce_catalog=cce_catalog, cce_dir=cce_dir
        )
    else:
        sim_info = ID
    sim = sim_info['sim']

    t0_array = sim.times[(sim.times > -30) & (sim.times < 100)]

    t0N_list = []
    for i, mm_list in enumerate(sim_info['mm_lists']):
        if i == 0:
            # Find the knee of the mismatch curve. The if statement keeps kneed
            # code from choosing the wrong knee
            kneedle = KneeLocator(
                t0_array,
                np.log(mm_list),
                S=10,
                curve='convex',
                direction='decreasing',
                online=True
            )
            zeroknee = kneedle.knee
        else:
            use = np.where(t0_array < zeroknee)
            kneedle = KneeLocator(
                t0_array[use],
                np.log(np.array(mm_list)[use]),
                S=10,
                curve='convex',
                direction='decreasing',
                online=True
            )

        t0N_list.append([kneedle.knee, kneedle.knee_y])

    sim_info['t0M_list'] = t0N_list
    return sim_info


def generate_epsilon_curve(
        ID: int,
        Nmax: int,
        cce_catalog: list = cce_catalog,
        cce_dir: str = cce_dir,
        additional_modes=[]
):
    """
    Generate the epsilon curves for each N up to Nmax for a given simulation.

    Parameters
    ----------
    ID : int OR dict
        The ID number of the simulation or the simulation info as a dictionary
    Nmax : int
        The maximum number of overtones to be fitted.
    cce_catalog : list (optional)
        List of dictionaries with properties of each CCE sim.
    cce_dir : str (optional)
        The directory in which the CCE data can be found.
    additional_modes : list (optional)
        Any additional modes which should be added to the simulation. Default
        is for none to be added.

    Returns
    -------
    sim_info : dictionary
        Keys:
            'name' : str
                The simulation name.
            'preferred_R' : float
                The preferred radius for the simulation.
            'url' : str
                Link to the simulation on zenodo.org
            'h' : ndarray
                A time series of the strain.
            'times' : ndarray
                Array of the times the simulation took place over.
            'metadata' : dict
                Provides the metadata for the remnant
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
            'eps_list' : ndarray
                Array of the epsilons at each timestep.
    """
    if type(ID) is int:
        sim_info = load_cce_data(ID, cce_catalog, cce_dir)
    else:
        assert type(ID) is dict, (
            "Check the stored value of ID, or input ID as an int to load data."
        )
        sim_info = ID
    sim = sim_info['sim']
    t0_array = sim.times[(sim.times > -30) & (sim.times < 80)]

    # Check whether epsilon has already been calculated
    if (
        'eps_lists' not in sim_info.keys() or
        ('eps_lists' in sim_info.keys() and len(sim_info['eps_lists']) < Nmax)
    ):
        sim_info['eps_lists'] = []
        mode_list = get_mode_list(Nmax, additional_modes)
        for i, modes in enumerate(mode_list):
            # Create epsilon curve
            print(f'\rCalculating for n = {i}')
            eps_list = []
            for t0 in t0_array:
                eps_list.append(
                    qnmfits.calculate_epsilon(
                        sim.times,
                        sim.h[2, 2],
                        modes,
                        Mf=sim.Mf,
                        chif=sim.chif_mag,
                        t0=t0,
                        t0_method='closest'
                    )[0]
                )
            sim_info['eps_lists'].append(eps_list)
    else:
        print("epsilon lists already calculated.")

    return sim_info


def t0NE_finder(
        ID,
        Nmax: int,
        cce_catalog: list = cce_catalog,
        cce_dir: str = cce_dir,
        prominence: float = .1
):
    """
    Finds the locations of t0N for a simulation, based on the minimum of the
    epsilon curve.

    Parameters
    ----------
    ID : int or dict
        The ID number of the simulation or the simulation info as a dictionary
    Nmax : int
        Number of overtones you want t0 for
    cce_catalog : list (optional)
        List of dictionaries with properties of each CCE sim.
    cce_dir : str (optional)
        The directory in which the CCE data can be found.

    Returns
    -------
    sim_info : dictionary
        Keys:
            'name' : str
                The simulation name.
            'preferred_R' : float
                The preferred radius for the simulation.
            'url' : str
                Link to the simulation on zenodo.org
            'h' : ndarray
                A time series of the strain.
            'times' : ndarray
                Array of the times the simulation took place over.
            'metadata' : dict
                Provides the metadata for the remnant
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
            'eps_list' : ndarray
                Array of the epsilons at each timestep.
            't0E_list' : list
                List of [t0N, eps(t0N)] for each N.
    """
    # Check whether the epsilon curve has already been calculated, calculates
    # it otherwise
    if (
        type(ID) is dict
        and 't0E_list' in ID.keys()
        and len(ID['t0E_list']) >= Nmax
    ):
        sim_info = ID
        return sim_info
    if not (
        type(ID) is dict
        and 'eps_lists' in ID.keys()
        and len(ID['eps_lists']) >= Nmax
    ):
        sim_info = generate_epsilon_curve(
            ID, Nmax, cce_catalog=cce_catalog, cce_dir=cce_dir
        )
    else:
        sim_info = ID
    sim = sim_info['sim']

    t0_array = sim.times[(sim.times > -30) & (sim.times < 80)]

    t0N_list = []
    for eps_list in sim_info['eps_lists']:

        peaks = find_peaks(
            -np.log(np.array(eps_list)), prominence=prominence
        )[0]

        # Find the actual minima, not the noise
        mins = peaks[np.array(eps_list)[peaks] < 0.01]
        t0N_list.append([t0_array[mins[0]], eps_list[mins[0]]])

    sim_info['t0E_list'] = t0N_list
    return sim_info


def injection(
        ID,
        N,
        additional_modes=[],
        returnC=False,
        tref=None
):
    """
    Generate a ringdown injection for a given simulation ID and number of
    overtones.

    Parameters
    ----------
    ID : int
        The ID number of the simulation.
    N : int
        The number of overtones to fit.
    additional_modes : list (optional)
        A list of additional modes to add to the fit. Default is an empty list.
    returnC : bool (optional)
        If True, return the C and omega values of the fit. Default is False.
    tref : float (optional)
        The time at which the fit is performed. If None, the t0 as determined
        by the mismatch curves will be used.

    Returns
    -------
    injection_info : dict
        A dictionary containing the injection information. Keys:
            'name' : str
                The name of the injection.
            'metadata' : dict
                The metadata for the remnant.
            'sim' : qnmfits.Custom object
                The simulation as a qnmfits object.
    C : ndarray (optional)
        The C values of the fit, if returnC is True.
    omega : ndarray (optional)
        The omega values of the fit, if returnC is True.
    """

    # Get the simulation data
    sim_info = load_cce_data(ID)
    sim = sim_info['sim']

    # List of QNMs to fit
    modes = [(2, 2, n, 1) for n in range(N+1)] + additional_modes

    # If tref is None, use the t0 as determined by the mismatch curves
    if tref is None:
        sim_info = t0NM_finder(
            sim_info, N,
        )
        t0 = sim_info['t0_list'][N][0]
    else:
        t0 = tref

    # Perform the fit
    best_fit = qnmfits.ringdown_fit(
        sim.times,
        sim.h[2, 2],
        modes,
        Mf=sim.Mf,
        chif=sim.chif_mag,
        t0=t0,
        t0_method='closest'
    )

    injection = best_fit['model']
    injection_times = best_fit['model_times']
    C = best_fit['C']
    omega = best_fit['frequencies']

    injection_dict = {}
    for m in range(-2, 2+1):
        if m == 2:
            injection_dict[(2, 2)] = injection
        else:
            injection_dict[(2, m)] = np.zeros_like(injection)

    injection_info = {}
    injection_info['name'] = f'Injection_{ID}_N{N}' + ''.join(
        [f'_{m[0]}{m[1]}{m[2]}{m[3]}' for m in additional_modes]
    )

    sim = qnmfits.Custom(
        injection_times,
        injection_dict,
        sim_info['metadata'],
        # zero_time=(2, 2)
    )

    injection_info['metadata'] = sim_info['metadata']
    injection_info['sim'] = sim

    if not returnC:
        return injection_info
    else:
        return injection_info, C, omega
