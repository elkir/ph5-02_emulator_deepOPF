import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
from matplotlib.patches import RegularPolygon

#####################################
#### Plots
#########################
#### Hex Plot
################



# def custom_plot(x, y, ax=None, **plt_kwargs):
#     if ax is None:
#         ax = plt.gca()
#     ax.plot(x, y, **plt_kwargs) ## example plot here
#     return(ax)
# def multiple_custom_plots(x, y, ax=None, plt_kwargs={}, sct_kwargs={}):
#     if ax is None:
#         ax = plt.gca()
#     ax.plot(x, y, **plt_kwargs) #example plot1
#     ax.scatter(x, y, **sct_kwargs) #example plot2
#     return(ax)

def hex_plot(X,
        labels = ['C', 'N0', 'N1', 'N2', 'N3', 'N4', 'N5'],
        alpha=0.8,
        vmax=1,
        ax=None, **hex_kwargs):

    if ax is None:
        ax = plt.gca()

    # Define color map
    newcolors = matplotlib.cm.get_cmap('coolwarm',128)
    norm = matplotlib.colors.TwoSlopeNorm(0,-vmax,vmax)
    color_mapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=newcolors)

    # Converting hex coordinates to cartesian
    coord = [[0,0,0],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0],[1,0,-1]]
    hcoord = [c[0] for c in coord]
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]
    
    colors = color_mapping.to_rgba(X)
    labels = [[x] for x in labels]

    # Add some coloured hexagons
    for x, y, c, l in zip(hcoord, vcoord, colors, labels):
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3., 
                            orientation=np.radians(30), 
                            facecolor=c, alpha=alpha, edgecolor='k', **hex_kwargs)
        ax.add_patch(hex)
        
        # Also add a text label
        ax.text(x, y, l[0], ha='center', va='center', size=20)

    # # Also add scatter points in hexagon centres
    # ax.scatter(hcoord, vcoord, c=colors, alpha=0.5)

    ax.set_xlim((-11/6, 11/6))
    ax.set_ylim((-11/6, 11/6))
    ax.set_aspect('equal')

    ax.set_frame_on(False)
    ax.set_axis_off()
    return(ax)




class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
def print_t(dfs):
    print(f"{color.BOLD}Timeseries:{color.END}")
    for k in (dfs.keys()):
        strEmpty = "Empty"  if dfs[k].empty else "Not-Empty"
        print(f"{k:<20} {strEmpty}")
        
### Histogram helper fun
def histOutline(dataIn, *args, **kwargs):
    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (data, bins)
def print_h(str):
    print(f"{color.BOLD}{color.UNDERLINE}{color.PURPLE}{str}{color.END}")

def network_summary(n):    
    print_h("Time range")
    s = n.snapshots
    date_min = s.min()
    date_max = s.max()
    print("Min: ",date_min)
    print("Max: ",date_max)
    print("Range: ",s.max()- s.min())
    print("Frequency: ",s.inferred_freq)
    print("Shape: ",s.shape)
    print_h("Buses")
    display(n.buses.head())
    print(f"{color.BOLD}{color.YELLOW}{n.buses.shape[0]} buses{color.END}")
    print_t(n.buses_t)
    print_h("Generators")
    display(n.generators.head())
    print(f"{color.BOLD}{color.YELLOW}{n.generators.shape[0]} generators{color.END}")
    print_t(n.generators_t)
    n.generators_t.p_max_pu.head()
    print_h("Loads")
    display(n.loads.head())
    print(f"{color.BOLD}{color.YELLOW}{n.loads.shape[0]} generators{color.END}")
    print_t(n.loads_t)
    print_h("Lines and links")
    display(n.lines.head())
    display(n.links.head())
    print(f"{color.BOLD}{color.YELLOW}{n.lines.shape[0]} lines and {n.links.shape[0]} links{color.END}")
    print_t(n.lines_t)
    print_t(n.links_t)
    
def index_of_full_dfs(k,n):
    # check network atribute that is a collection of DFs and return the non-empty ones
    dfs = getattr(n,k)
    a=[]
    for k in (dfs.keys()):
        if not dfs[k].empty:
             a.append(k)
    return a

    
def network_get_nonempty_timeseries(n):
    keys = n.__dict__.keys()
    keys_t = [k for k in keys if "_t" == k[-2:]]
    index = [(k,i) for k in keys_t for i in index_of_full_dfs(k,n)]
    return index # [(attribute, df_name),...]


def n_extract_values(n, index):
    '''
    Extracts timeseries specified in index (list of (atribute,column,Nvars) pairs)
    Adds a prefix to column names as "attr:col: <name>"
    ''' 
    return pd.concat([getattr(n,c)[d].add_prefix(f"{c[:-2]}:{d}: ") for c,d,_ in index],axis=1)
