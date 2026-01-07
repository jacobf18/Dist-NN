"""
Configuration settings for matplotlib and plotting.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

def setup_plotting():
    """Configure matplotlib settings for publication-quality plots."""
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    plt.rc('axes', axisbelow=True)
    label_size = 20
    mpl.rcParams['text.usetex'] = True 
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = label_size
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
    pylab.rcParams['xtick.major.pad'] = 5
    pylab.rcParams['ytick.major.pad'] = 5

# Line styles and markers for plotting
LINE_STYLES = ['--', ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
MARKERS = ['>', 'o', 's', 'D', '>', 's', 'o', 'D', '>', 's', 'o', 'D']

