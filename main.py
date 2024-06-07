# from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
import datetime
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import networkx as nx
from matplotlib import animation

from config import *
import cmasher as cmr

import networkx as nx

PROGRAM_START = datetime.datetime.now()

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 19
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def create_empty_frames(resolution, overwrite=False):
    df = pd.DataFrame(columns=list(range(resolution)), index=list(range(resolution)))
    for i in range(FROM_INDEX, TO_INDEX):
        path = f"Time Frames\\{resolution}_{i}.csv"
        if not os.path.isfile(path) or overwrite:
            df.to_csv(path, index=False)


# Process raw tensor into csv files for each time point
def process_raw():
    print(f"Processing raw data from indices {FROM_INDEX} to {TO_INDEX}")
    create_empty_frames(256)
    for cnum, chunk in enumerate(pd.read_csv(RAW_PATH, header=None, chunksize=CHUNKSIZE, dtype=np.float32)):
        for tp, col in ((i, chunk[i]) for i in range(FROM_INDEX, TO_INDEX)):
            path = f"Time Frames\\256_{tp}.csv"
            df = pd.read_csv(path)
            for rnum, v in col.items():
                df.iloc[rnum % 256].iloc[rnum // 256] = v
            df.to_csv(path, index=False)
        print(f"Progress: {round(cnum * CHUNKSIZE * 100 / 256 ** 2, 1)}%", end="")
        print(f"\t Elapsed Time: {datetime.datetime.now() - PROGRAM_START}")


def generate_png(tp):
    frame_path = f"Time Frames\\256_{tp}.csv"
    image_path = f"Images\\256_{tp}.png"

    df = pd.read_csv(frame_path)

    fig, ax = plt.subplots()
    heatmap = ax.imshow(df, vmin=-3, vmax=3, cmap=cmr.wildfire)

    cbar = plt.colorbar(heatmap)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

    plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, facecolor="black")
    plt.close()


def batch_image():
    for i in range(FROM_INDEX, TO_INDEX):
        generate_png(i)

        if i % 10 == 0:
            print(f"Progress: {round((i - FROM_INDEX) * 100 / (TO_INDEX - FROM_INDEX), 1)}%", end="")
            print(f"\t Elapsed Time: {datetime.datetime.now() - PROGRAM_START}")


def make_gif(image_directory, tp_i, tp_num, fps, name):
    # Create a list of image file paths
    image_files = [f"{image_directory}\\{tp_i + i}.png" for i in range(0, tp_num)]

    img = plt.imread(image_files[0])
    h, w, _ = img.shape

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))

    # Initialize an empty list to store the image frames
    frames = []

    # Iterate over the image files and append each image to the frames list
    for image_file in image_files:
        img = plt.imread(image_file)
        frame = plt.imshow(img, animated=True, extent=[0, w, 0, h])
        plt.axis('off')
        frames.append([frame])

    # Create the animation
    animation_object = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True, repeat_delay=0)

    animation_object.save(f'Animations\\{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{name}.gif',
                          fps=fps, writer='ffmpeg')


class Node:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    cluster = None


class Cluster:
    root: Node = None
    nodes: set[Node] = set()


class Graph:
    nodes = [[Node(i, j) for i in range(RESOLUTION // SCALE)] for j in range(RESOLUTION // SCALE)]
    edges = set()


def make_graphs2():
    # Initialize empty graph with no clusters
    graph = Graph()

    tp_sizes = {}

    # Iterate through the dataframes of each time point
    # Seizure?: 6300-6500

    for tp in range(6000, 6010): #6100 #6300
        print(tp)
        frame_path = f"Time Frames\\256_{tp}.csv"
        df = pd.read_csv(frame_path)

        arr = np.array(df)
        std = np.std(arr[arr != 0])

        arr = np.where(arr > std, 1.0, 0.0)

        clusters: list[list[(int, int)]] = []

        for i, j in np.ndindex(arr.shape):
            if arr[i,j] == 1:
                stack = [(i, j)]
                cluster = []

                while stack:
                    x, y = stack.pop()
                    arr[x, y] = 0
                    cluster.append((x, y))

                    # Check neighbors
                    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    for nx, ny in neighbors:
                        if arr[nx, ny] == 1:
                            stack.append((nx, ny))

                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)

        v = 0.95

        sizes = []

        for cluster in clusters:
            for i, j in cluster:
                arr[i, j] = v
            v -= 0.1
            if v < 0:
                v = 0.95
            sizes.append(len(cluster))

        tp_sizes[tp] = sizes

        image_path = f"Clusters\\256_{tp}.png"

        fig, ax = plt.subplots()
        heatmap = ax.imshow(arr, vmin=0, vmax=1, cmap="nipy_spectral")

        cbar = plt.colorbar(heatmap)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1, facecolor="black")
        plt.close()

    #make_gif("Clusters\\")

    sizes = []

    for tp, v in tp_sizes.items():
        sizes += v

    # Compute logarithmically scaled bins
    log_bins = np.logspace(0, 6, 30)

    # Create histogram with logarithmically scaled bins
    counts, bins = np.histogram(sizes, bins=log_bins)

    # Compute log of bin centers and counts
    log_bin_centers = (bins[1:] + bins[:-1]) / 2
    log_counts = np.log(counts)

    # Plot line graph of log of count vs log of cluster size
    fig, ax = plt.subplots()#figsize=(8, 6)
    ax.scatter(log_bin_centers, counts, color="black")#, marker='o', linestyle='-')
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('Count')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.show()
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-pow.png")


def make_graphs3():
    # Initialize empty graph with no clusters
    graph = Graph()

    tp_sizes = {}

    # Iterate through the dataframes of each time point
    # Seizure?: 6300-6500

    plt.rcParams["font.size"] = 17

    for tp in range(6300, 6500): #6100 #6300
        print(tp)
        frame_path = f"Time Frames\\256_{tp}.csv"
        df = pd.read_csv(frame_path)
        dat = df.copy()

        arr = np.array(df)
        std = np.std(arr[arr != 0]) * 3

        arr = np.where(np.abs(arr) > std, 1.0, 0.0)

        clusters: list[list[(int, int)]] = []

        for i, j in np.ndindex(arr.shape):
            if arr[i,j] == 1:
                stack = [(i, j)]
                cluster = []

                while stack:
                    x, y = stack.pop()
                    arr[x, y] = 0
                    cluster.append((x, y))

                    # Check neighbors
                    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                    for nx, ny in neighbors:
                        if arr[nx, ny] == 1:
                            stack.append((nx, ny))

                clusters.append(cluster)

        clusters.sort(key=lambda x: len(x), reverse=True)

        v = 0.95

        sizes = []

        for cluster in clusters:
            for i, j in cluster:
                arr[i, j] = v
            v -= 0.15
            if v < 0:
                v = 0.95
            sizes.append(len(cluster))

        tp_sizes[tp] = sizes

        image_path = f"Clusters\\256_{tp}.png"

        fig, axs = plt.subplots(1, 2)
        heatmap = axs[1].imshow(arr, vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")
        heatmap = axs[0].imshow(dat, vmin=-3, vmax=3, cmap=cmr.wildfire, interpolation="nearest")

        #axs[0].set_aspect("equal")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(r'Raw Image')

        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title(r'Avalanches')

        #cbar = plt.colorbar(heatmap)
        #cbar.ax.yaxis.set_tick_params(color="white")
        #cbar.outline.set_edgecolor("white")
        #plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        #plt.show()
        plt.savefig(image_path, bbox_inches="tight", dpi=200)#, pad_inches=0.1, facecolor="black")
        plt.close()

    plt.rcParams["font.size"] = 19

    #make_gif("Clusters\\")

    sizes = []

    for tp, v in tp_sizes.items():
        sizes += v

    # Compute logarithmically scaled bins
    log_bins = np.logspace(0, round(np.log(max(sizes)) / np.log(10)), 40)

    # Create histogram with logarithmically scaled bins
    counts, bins = np.histogram(sizes, bins=log_bins)

    # Compute log of bin centers and counts
    log_bin_centers = (bins[1:] + bins[:-1]) / 2

    x = np.log(log_bin_centers)
    y = np.log(counts)

    x = x[y > 0]
    y = y[y > 0]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    tr = log_bin_centers**slope * np.exp(intercept)

    # Plot line graph of log of count vs log of cluster size
    fig, ax = plt.subplots()#figsize=(8, 6)
    ax.scatter(log_bin_centers, counts, color="black", marker='x')#, linestyle='-')
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([tr[-1], counts[0] * 3])
    #plt.show()
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-pow.png", bbox_inches="tight", dpi=200)

    #plt.show()
    plt.close()

    # Plot the equation of the regression line
    eq1 = r'$\mathdefault{y=' + f"{np.exp(intercept):.1f}" + r'x^{' + f"{slope:.2f}" + r'\pm' + f"{std_err:.2f}" + r'}}$'
    eq2 = r'$\mathdefault{R^2=' + f"{r_value ** 2:.2f}" + r'}$'

    fig, ax = plt.subplots()  # figsize=(8, 6)
    ax.plot(log_bin_centers, tr, color="red", label="Fit")  # , marker='o', linestyle='-')
    ax.scatter(log_bin_centers, counts, color="black", marker='x')#, linestyle='-')
    ax.text(log_bin_centers[-1], tr[0], r'{}'.format(eq1), fontsize=19, color='black', horizontalalignment='right',
        verticalalignment='top')
    ax.text(log_bin_centers[1], tr[-1], r'{}'.format(eq2), fontsize=19, color='black')
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-fit.png", bbox_inches="tight", dpi=200)
    #plt.show()





class GraphClusterTracker:
    def __init__(self):
        self.graph = nx.Graph()
        self.clusters = []

    def add_edges(self, edges):
        self.graph.add_edges_from(edges)
        self.update_clusters()

    def update_clusters(self):
        self.clusters = list(nx.connected_components(self.graph))

    def track_cluster_changes(self, new_edges):
        old_clusters = self.clusters.copy()
        self.add_edges(new_edges)
        new_clusters = self.clusters
        changes = {}

        for old_cluster in old_clusters:
            for new_cluster in new_clusters:
                if old_cluster.isdisjoint(new_cluster):
                    continue
                intersect = old_cluster.intersection(new_cluster)
                removed_nodes = old_cluster - intersect
                added_nodes = new_cluster - intersect
                changes[tuple(intersect)] = {'removed_nodes': removed_nodes, 'added_nodes': added_nodes}

        return changes


def temporal_clusters():
    tp_i = 6000
    frames = 1000

    clusters: list[dict[int, int]] = []
    sizes = []

    raw_data_tensor = np.empty(shape=(frames, 256, 256), dtype=float)
    activation_tensor = np.empty(shape=(frames, 256, 256), dtype=float)

    for tp in range(0, frames):
        frame_path = f"Time Frames\\256_{tp_i + tp}.csv"
        df = pd.read_csv(frame_path)
        raw_data_tensor[tp] = np.array(df)
        std = np.std(raw_data_tensor[tp][raw_data_tensor[tp] != 0])
        activation_tensor[tp] = np.where(np.abs(raw_data_tensor[tp]) > 3 * std, 1.0, 0.0) # np.max(np.abs(raw_data_tensor[tp])) / 2

    color = 0.95

    for tp, i, j in np.ndindex(activation_tensor.shape):
        if activation_tensor[tp, i, j] == 1:
            stack = [(tp, i, j)]
            cluster = {}
            footprint = np.zeros((256, 256))

            while stack:
                t, x, y = stack.pop()
                activation_tensor[t, x, y] = color
                footprint[x, y] = 1

                try:
                    cluster[t] += 1
                except KeyError:
                    cluster[t] = 1

                # Check neighbors
                neighbors = [(t, x + 1, y),
                             (t, x - 1, y),
                             (t, x, y + 1),
                             (t, x, y - 1),
                             # (t - 1, x, y),
                             (t + 1, x, y)]

                for nt, nx, ny in neighbors:
                    if nt < 0 or nt >= frames or x < 0 or x >= 256 or y < 0 or y >= 256:
                        continue
                    if activation_tensor[nt, nx, ny] == 1:
                        activation_tensor[nt, nx, ny] = 0
                        stack.append((nt, nx, ny))

            clusters.append(cluster)
            sizes.append(np.sum(footprint))

            color -= 0.15
            if color < 0:
                color = 0.95

    plt.rcParams["font.size"] = 17

    # fig, axs = plt.subplots(1, 3)

    # axs[0].imshow(activation_tensor[4], vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")
    # axs[1].imshow(activation_tensor[5], vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")
    # axs[2].imshow(activation_tensor[6], vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")
    #
    # axs[0].set_xticks([])
    # axs[0].set_yticks([])
    # axs[0].set_title(r'1')
    #
    # axs[1].set_xticks([])
    # axs[1].set_yticks([])
    # axs[1].set_title(r'2')
    #
    # axs[2].set_xticks([])
    # axs[2].set_yticks([])
    # axs[2].set_title(r'3')

    # image_path = f"Plots\\Evolution.png"
    # plt.savefig(image_path, bbox_inches="tight", dpi=200)  # , pad_inches=0.1, facecolor="black")
    # plt.close()

    for tp in range(0, frames):
        image_path = f"TemporalClusters\\{tp_i + tp}.png"

        fig, ax = plt.subplots()
        ax.imshow(activation_tensor[tp], vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(image_path, bbox_inches="tight", dpi=200, pad_inches=0.1, facecolor="black")
        plt.close()

        image_path = f"Combined\\{tp_i + tp}.png"

        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(raw_data_tensor[tp], vmin=-3, vmax=3, cmap=cmr.wildfire, interpolation="nearest")
        axs[1].imshow(activation_tensor[tp], vmin=0, vmax=1, cmap="nipy_spectral", interpolation="nearest")

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(r'Recording')

        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title(r'Avalanches')

        plt.savefig(image_path, bbox_inches="tight", dpi=200)#, pad_inches=0.1, facecolor="black")
        plt.close()

    make_gif("TemporalClusters", tp_i, frames, 10, "single")
    make_gif("Combined", tp_i, frames, 10, "combined")

    plt.rcParams["font.size"] = 19

    size, count = np.unique(sizes, return_counts=True)

    size = size[count != 1]
    count = count[count != 1]

    x = np.log(size)
    y = np.log(count)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    tr = size**slope * np.exp(intercept)

    # Plot line graph of log of count vs log of cluster size
    fig, ax = plt.subplots()
    ax.scatter(size, count, color="black", marker="x")
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([tr[-1], count[0] * 3])
    plt.tight_layout()

    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-sizes.png", bbox_inches="tight",
                dpi=200)
    plt.close()

    # Plot the equation of the regression line
    eq1 = r'$\mathdefault{y=' + f"{np.exp(intercept):.1f}" + r'x^{' + f"{slope:.2f}" + r'\pm' + f"{std_err:.2f}" + r'}}$'
    eq2 = r'$\mathdefault{R^2=' + f"{r_value ** 2:.2f}" + r'}$'

    fig, ax = plt.subplots()
    ax.plot(size, tr, color="red", label="Fit", linestyle="dashed")
    ax.scatter(size, count, color="black", marker='x')
    ax.text(size[-1], tr[0], r'{}'.format(eq1), fontsize=19, color='black', horizontalalignment='right',
            verticalalignment='top')
    ax.text(size[0], tr[-1], r'{}'.format(eq2), fontsize=19, color='black')
    ax.set_xlabel('Avalanche Size')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-sizes-fit.png", bbox_inches="tight",
                dpi=200)

    durations = [len(d) for d in clusters]

    size, count = np.unique(durations, return_counts=True)

    size = size[count != 1]
    count = count[count != 1]

    x = np.log(size)
    y = np.log(count)

    # Perform linear regression
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    tr = size ** slope * np.exp(intercept)

    # Plot line graph of log of count vs log of cluster size
    fig, ax = plt.subplots()
    ax.scatter(size, count, color="black", marker="x")
    ax.set_xlabel('Avalanche Length')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([tr[-1], count[0] * 3])
    plt.tight_layout()

    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lens.png", bbox_inches="tight",
                dpi=200)
    plt.close()

    # Plot the equation of the regression line
    eq1 = r'$\mathdefault{y=' + f"{np.exp(intercept):.1f}" + r'x^{' + f"{slope:.2f}" + r'\pm' + f"{std_err:.2f}" + r'}}$'
    eq2 = r'$\mathdefault{R^2=' + f"{r_value ** 2:.2f}" + r'}$'

    fig, ax = plt.subplots()
    ax.plot(size, tr, color="red", label="Fit", linestyle="dashed")
    ax.scatter(size, count, color="black", marker='x')
    ax.text(size[-1], tr[0], r'{}'.format(eq1), fontsize=19, color='black', horizontalalignment='right',
            verticalalignment='top')
    ax.text(size[0], tr[-1], r'{}'.format(eq2), fontsize=19, color='black')
    ax.set_xlabel('Avalanche Length')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lens-fit.png", bbox_inches="tight",
                dpi=200)

    durations_dict = {}
    for i, v in enumerate(durations):
        try:
            durations_dict[v].append(sizes[i])
        except KeyError:
            durations_dict[v] = [sizes[i]]

    durations_dict = {k: np.mean(l) for k, l in durations_dict.items() if len(l) > 1}
    durations_dict = dict(sorted(durations_dict.items()))

    durs = list(durations_dict.keys())
    sizes = list(durations_dict.values())

    x = np.log(durs)
    y = np.log(sizes)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    tr = durs ** slope * np.exp(intercept)

    # Plot line graph of log of count vs log of cluster size
    fig, ax = plt.subplots()
    ax.scatter(durs, sizes, color="black", marker="x")
    ax.set_xlabel('Avalanche Length')
    ax.set_ylabel('Average Avalanche Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_ylim([tr[-1], sizes[0] * 3])
    plt.tight_layout()

    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lens-sizes.png", bbox_inches="tight",
                dpi=200)
    plt.close()

    # Plot the equation of the regression line
    eq1 = r'$\mathdefault{y=' + f"{np.exp(intercept):.1f}" + r'x^{' + f"{slope:.2f}" + r'\pm' + f"{std_err:.2f}" + r'}}$'
    eq2 = r'$\mathdefault{R^2=' + f"{r_value ** 2:.2f}" + r'}$'

    fig, ax = plt.subplots()
    ax.plot(durs, tr, color="red", label="Fit", linestyle="dashed")
    ax.scatter(durs, sizes, color="black", marker='x')
    ax.text(durs[0], tr[-1], r'{}'.format(eq1), fontsize=19, color='black', verticalalignment='top')
    ax.text(durs[-1], tr[0], r'{}'.format(eq2), fontsize=19, color='black', horizontalalignment='right')
    ax.set_xlabel('Avalanche Length')
    ax.set_ylabel('Average Avalanche Size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lens-sizes-fit.png", bbox_inches="tight",
                dpi=200)
    plt.close()


def make_graphs():
    # Iterate through the dataframes of each time point

    edges = []

    for tp in range(6153, 6154):
        frame_path = f"Time Frames\\256_{tp}.csv"
        df = pd.read_csv(frame_path)

        # Create GraphClusterTracker instance
        tracker = GraphClusterTracker()

        for i in reversed(range(1, RESOLUTION, 4)):
            for j in reversed(range(1, RESOLUTION, 4)):

                nv = df.iloc[i, j]
                tv = df.iloc[i-1, j]
                lv = df.iloc[i, j-1]

                if abs((nv - tv) / nv) <= TOL:
                    edges.append((i-1, j))
                if abs((nv - lv) / nv) <= TOL:
                    edges.append((i, j-1))


        # Initial set of edges
        initial_edges = edges
        tracker.add_edges(initial_edges)

        # Identify clusters
        print("Initial clusters:", tracker.clusters)

        # New set of edges
        new_edges = [(3, 4), (5, 6)]
        changes = tracker.track_cluster_changes(new_edges)

        # Track changes in clusters
        print("Changes in clusters:")
        for cluster, change in changes.items():
            print(f"Cluster {cluster}: Removed nodes {change['removed_nodes']}, Added nodes {change['added_nodes']}")


def ising():
    # Example array of spins (randomly generated)
    size = 5

    X, Y = np.meshgrid(np.arange(size+1), np.arange(size))
    U = np.zeros((size+1, size))
    V = np.ones((size, size+1)) * -1
    V[:, -1] = [5]*size

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 19
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    fig, axs = plt.subplots(1, 3)

    axs[0].quiver(X, Y, U, V, V, pivot="mid", scale=8, width=0.02, headwidth=3, cmap="hot")
    axs[0].set_aspect("equal")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlim([-0.5, size-0.5])
    axs[0].set_ylim([-0.5, size-0.5])
    axs[0].set_title(r'$\mathdefault{T < T_C}$', pad=10)

    V = np.array([[1, -1, -1, -1,- 1, 5],
                  [1, 1, 1, -1, -1, 5],
                  [1, 1, 1, -1, -1, 5],
                  [-1, 1, 1, 1, -1, 5],
                  [-1, -1, 1, 1, 1, 5]])

    axs[1].quiver(X, Y, U, V, V, pivot="mid", scale=8, width=0.02, headwidth=3, cmap="hot")
    axs[1].set_aspect("equal")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlim([-0.5, size-0.5])
    axs[1].set_ylim([-0.5, size-0.5])
    axs[1].set_title(r'$\mathdefault{T = T_C}$', pad=10)

    V = np.random.choice([-1, 1], size=(size, size+1))
    V[:, -1] = [5]*size

    axs[2].quiver(X, Y, U, V, V, pivot="mid", scale=8, width=0.02, headwidth=3, cmap="hot")
    axs[2].set_aspect("equal")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_xlim([-0.5, size - 0.5])
    axs[2].set_ylim([-0.5, size - 0.5])
    axs[2].set_title(r'$\mathdefault{T > T_C}$', pad=10)

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"Plots\\{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Ising.png", bbox_inches="tight", dpi=200)#, pad_inches=0.1, facecolor="black")


def main():
    temporal_clusters()
    #make_graphs3()
    #ising()


if __name__ == "__main__":
    main()

    print("Program completed.")


'''sam_df = pd.read_csv(PATH, header=None, skiprows=40000, nrows=30)

time = []
avg = []

print(1)

t = 0
for col in sam_df:
    time.append(t)
    avg.append(abs(sam_df[col].mean()))

    t += 0.02

plt.plot(time, avg)
plt.show()


for i in range(5, resolution):
    for j in range(0, resolution):
        print(f"{i}, {j}")

        r = 8192 * i + 32 * j
        data = pd.read_csv(PATH, header=None, skiprows=r, nrows=1)

        for k, df in enumerate(timepoints):
            df.iloc[i].iloc[j] = data.iloc[0].iloc[6500 + (k * 2)]
            if j == resolution - 1:
                df.to_csv(f"t2\\{k}.csv", index=False)




one_row_df = pd.read_csv("oneframe.csv").transpose()

frame = pd.DataFrame(columns=[i for i in range(256)])

for i in range(256):
    row = one_row_df.iloc[0].iloc[i * 256:(i + 1) * 256].to_frame().reset_index(drop=True).transpose()

    frame = pd.concat([frame, row])


fig, ax = plt.subplots()
heatmap = ax.imshow(frame, cmap='magma')

plt.show()
'''