# this script computes clusters from utterance time points
# via curve fitting and a slope difference algorithm
# curve fitting/slope diff equations are adapted from Andreas Højlund's matlab implementation

# it needs to be called from another script; and the 'run' function takes
# a filepath to a csv transcript file, then returns the entire file clustered

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# defining the curve function
def func(x, c, m):
    return c * (1 - np.exp(-m * x))

# computing clusters
def run(filepath):

    # reading the data
    data = pd.read_csv(filepath, sep=";", encoding="latin-1")

    # extracting unique filenames - these will be used to create subsets
    subset_list = data["Filename"].unique()

    temp_data_list = []
    skipped_files = []

    # looping through the subset list
    for i in range(len(subset_list)):

        # constructing a subset with only the first task
        subset = data[data["Filename"] == subset_list[i]]

        # constructing x and y vectors
        xdata = subset['Utterance start time'].values.copy()
        ydata = np.arange(1, len(xdata) + 1)

        # fixing x = 0 issue
        xdata[xdata == 0] = 0.000001

        # fitting the curve using sci-pi
        try:
            popt, pcov = curve_fit(func, xdata, ydata, method='lm')
        except RuntimeError as e:
            print(f"Skipping {subset_list[i]}: {e}")
            skipped_files.append(subset_list[i])
            continue

        # implementing the slope difference algorithm used in Andreas Højlunds matlab script
        # first, we compute the distance in seconds between utterance timestamps
        xdiffs = np.diff(xdata)

        # then, we compute the difference between the observed utterance distance and the distance predicted by the curve
        xclus = 1/xdiffs - np.diff(func(xdata, c=popt[0], m=popt[1]))/xdiffs

        # finally, we determine cluster status
        xclusidx = xclus > 0 

        # adding 'false' to the first dot
        xclusidx = np.concatenate([[False], xclusidx])

        # creating cluster IDs
        cluster_bin = xclusidx.astype(int).copy()
        current_cluster = 0

        for t in range(len(cluster_bin) - 1):
            if cluster_bin[t+1] == 1 and cluster_bin[t] == 0:
                current_cluster += 1
                cluster_bin[t] = current_cluster
            elif cluster_bin[t] == 1:
                cluster_bin[t] = current_cluster

        # handle the last element
        if cluster_bin[-1] == 1:
            cluster_bin[-1] = current_cluster

        cluster_ID = cluster_bin

        # appending the cluster information to the time points
        subset['Cluster Status'] = xclusidx.astype(int)
        subset['Cluster ID'] = cluster_ID
        subset['curve_c'] = popt[0] # appending curve parameters
        subset['curve_m'] = popt[1]

        temp_data_list.append(subset)

        # plotting points on curve colored by cluster ID
        plt.figure()
        n_clusters = subset['Cluster ID'].max()

        colors_list = ['grey'] + list(plt.cm.turbo(np.linspace(0, 1, n_clusters)))
        cmap = mcolors.ListedColormap(colors_list)
        bounds = np.arange(-0.5, n_clusters + 1.5)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.plot(xdata, func(xdata, *popt), 'k--', label='fit', linewidth=0.8)
        scatter = plt.scatter(xdata, ydata, c=cluster_bin, cmap=cmap, norm=norm, s=20, zorder=5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative utterance count')
        plt.title(f'Fitting curve to utterances from file {subset_list[i]}')

        cbar = plt.colorbar(scatter, ticks=np.arange(0, n_clusters + 1))
        cbar.set_ticklabels(['no cluster'] + [f'cluster {i}' for i in range(1, n_clusters + 1)])

        plt.savefig(f'/work/verbal_fluency/curvefit/{subset_list[i]}.png')

    # unpack the list into a single data frame
    processed_data = pd.concat(temp_data_list, axis=0, ignore_index=True)

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} file(s): {skipped_files}")

    # returns the processed data
    return processed_data