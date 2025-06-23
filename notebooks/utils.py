import numpy as np
import pandas as pd

import json
import pickle
import qnmfits
import copy
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
        cce_dir: str = cce_dir
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
        cce_dir / f"{sim_info['name']}" / 'Lev5' / data_filename, 'rb'
    ) as f:
        h = pickle.load(f)

    # Load the remnant properties
    metadata_filename = (
        f"metadata_BondiCce_R{int(sim_info['preferred_R']):04d}"
        "_superrest.json"
    )
    with open(
        cce_dir / f"{sim_info['name']}" / 'Lev5' / metadata_filename, 'r'
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
            't0_list' : list
                List of of [t0N, MM(t0N)] for each N.
    """
    # Check whether the mismatch curve has already been calculated, calculates
    # it otherwise
    if (
        type(ID) is dict
        and 't0_list' in ID.keys()
        and len(ID['t0_list']) >= Nmax
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

    sim_info['t0_list'] = t0N_list
    return sim_info


sim_path = cce_dir / 'sim_info_allN.pkl'
with open(sim_path,'rb') as fp:
    sim_info_allN = pickle.load(fp)


def t0NE_finder(path:str, prominence:float = .1, return_y = True):
    """Finds the values of t0_E from the file path for a dataset."""
    # Blank lists to add to
    t0NE=[]
    epsilons = []

    #read in data and rget rid of pandas column number artifact
    data = pd.read_csv(path)
    del data['Unnamed: 0']

    # make the data keys numbers instead of strings
    keys_float=[]
    for key in data.keys():
        try:
            keys_float.append(int(key))
        except ValueError:
            keys_float.append(key)
    data.columns=keys_float

    # Iterate through all keys
    for key in data.keys():

    # Avoid the t column
        if type(key)==str:
            continue
        
        peaks = find_peaks(-np.log(np.array(data[key])),prominence=prominence)[0]
        # Find the actual minima, not the noise
        mins = peaks[data[key][peaks]<0.01]
        t0NE.append(data['t'][mins[0]])
        # also return
        epsilons.append(data[key][mins[0]])
    if return_y:
        return [[i,j] for i,j in zip(t0NE, epsilons)]
    return t0NE


def injection(ID, N, additional_modes=[], data_location=sim_path, returnC = False, tref=None):

    """Create an injection waveform from N amplitudes with CCE ID."""

    try:
        sim_info=sim_info_allN[ID]
    except NameError:
        with open(data_location,'rb') as fp:
             sim_info_allN = pickle.load(fp)
        sim_info=sim_info_allN[ID]
    # generate the modes
    modes = [(2,2,n,1) for n in range(N+1)]+additional_modes
    sim=sim_info['sim']
    if tref is None:
        t0=sim_info['t0_list'][N][0]
    else:
        t0 = tref
    # Perform the fit
    best_fit = qnmfits.ringdown_fit(
        sim.times,
        sim.h[2,2],
        modes,
        Mf=sim.Mf,
        chif=sim.chif_mag,
        t0=t0
    )

    injection = copy.deepcopy(sim_info)
    injection_times = best_fit['model_times']
    t0=sim_info['t0_list'][N][0]
    T=100
    C = best_fit['C']
    omega = best_fit["frequencies"]

    for key in sim.h.keys():
        # Get rid of all modes except [2,2]
        del injection['sim'].h[key] 

    # Get the injection to 
    injection['sim'].h[2,2] = best_fit['model']
    injection['sim'].times = injection_times


    # Get rid of or change wrong/unuseful keys 
    del injection['url']
    injection['name']='Injection'
    del injection['mm_lists']
    del injection['t0_list']
    if not returnC:
        return injection 
    else:
        return injection, C, omega


def find_amplitude(ID:int, N_overtone:int, N_fit: int,sim_info_allN=sim_info_allN,t0N='t0N'):
    r"""Return the amplitude of the (2,2,N_overtone) QNM (adjusted to tref=0) 
    in a fit with N_fit overtones performed at t0 N_fit.
    Parameters
    ----------
    ID : int
        Identification number of the CCE waveform to be used for data
    N_overtone : int
        Function will return the amplitudes of the (2,2,N_overtone) QNM 
    N_fit : int (optional)
        Number of overtones to be used in the fit
    t0N : str or float (optional)
        Specifies Default is 't0N', in which case the fit time is used as t0_N. 
    Returns
    ----------
    amplitude : float

    """
    # Make sure simulation dictionary exists, and has the right data in it 
    try:
        sim_info=sim_info_allN[ID]
    except (KeyError):
        print("Make sure the variable sim_info_allN contains the data loaded from sim_info_allN.pkl.")
    assert 't0_list' in sim_info.keys(), "No t0 N data found."
    assert  len(sim_info['t0_list'])>=N_fit, f"Expected {N_fit} $t_0^N$ values, got {len(sim_info['t0_list'])}."
    assert N_overtone<=N_fit, "N_overtone must be less than or equal to N_fit"

    tref=0
    # Generate the list of modes to be used in the fit
    modes = [(2,2,n,1) for n in range(N_fit+1)]

    # Look up t0N, set variables for fit
    if t0N == "t0N":
        t0N = sim_info['t0_list'][N_fit][0]
    if t0N!='t0N':
        try:
            t0N = float(t0N)
        except TypeError:
            print("t0N must be a number.")
    sim = sim_info['sim']
    data = sim.h[2,2]
    times = sim.times
    Mf=sim.Mf
    chif=sim.chif_mag

    # Perform a fit with the given start time and modes
    best_fit = qnmfits.ringdown_fit(times, data, modes, Mf, chif, t0 = t0N)
    omega = best_fit['frequencies'][N_overtone]
    C = best_fit['C'][N_overtone]

    # Adjust to tref = 0
    adjust=np.exp(-1j*omega*(tref-t0N))
    amplitude= np.abs(C*adjust) 

    return amplitude
    
def find_delta_amplitude(ID:int,N_overtone:int,N_fit:int,sim_info_allN=sim_info_allN,t0N='t0N'):
    """Find Delta A_(2,2,N_overtone)^N_fit. """
    A220 = find_amplitude(ID, N_overtone, N_fit,sim_info_allN=sim_info_allN,t0N=t0N)
    A22N=find_amplitude(ID, N_overtone, N_overtone,sim_info_allN=sim_info_allN,t0N=t0N)
    return (A22N-A220)/A220


def sort_paths(list):
    """Puts all paths in a list in numerical order by file name. 
        This function is probably not necessary (and probably 
        could be executed more efficiently) but not having the paths 
        sorted frustrates me."""
    decorated=[]
    for item in list:
        start = item[::-1].index('/')
        for num,i  in enumerate(item[-start:]):
            try:
                n = int(item[-start:][:num])
            except ValueError:
                if num!=0:
                    break
        decorated.append((n,item))
    decorated.sort()
    return [item for n, item in decorated]  




