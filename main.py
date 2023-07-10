import glob
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation

from config import *
import cmasher as cmr

PROGRAM_START = datetime.datetime.now()


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


def make_gif(image_directory):
    # Create a list of image file paths
    image_files = sorted(glob.glob(os.path.join(image_directory, '*.png')))

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Initialize an empty list to store the image frames
    frames = []

    # Iterate over the image files and append each image to the frames list
    for image_file in image_files:
        img = plt.imread(image_file)
        frame = plt.imshow(img, animated=True)
        frames.append([frame])

    # Create the animation
    animation_object = animation.ArtistAnimation(fig, frames, interval=20, blit=True, repeat_delay=0)

    # Set the number of frames per second (optional)
    frames_per_second = 50
    animation_object.save('animation.gif', fps=frames_per_second, writer='ffmpeg')


def make_graphs():
    # Iterate through the dataframes of each time point
    for tp in range(FROM_INDEX, TO_INDEX):
        tp = 6153

        frame_path = f"Time Frames\\256_{tp}.csv"
        df = pd.read_csv(frame_path)

        # Hash dictionary of attributes to each node by int name
        nodes: dict[int, dict] = {k: {} for k in range(RESOLUTION**2)}

        # Track members of clusters by int name
        clusters: list[set] = []

        



        def cluster(node, adj):
            # Add to the adjacent node's cluster if clustered already
            if cluster := nodes[adj]["cluster"]:
                cluster.add(node)
                nodes[node]["cluster"] = cluster
            # Create a new cluster with both nodes if adjacent not clustered already
            else:
                cluster = {adj, node}
                clusters.append(cluster)
                nodes[node]["cluster"] = cluster
                nodes[adj]["cluster"] = cluster

        # Iterate through each node in grid
        for i in range(0, RESOLUTION):
            for j in range(0, RESOLUTION):
                node = i * RESOLUTION + j

                n_val = df.iloc[i, j]

                # attr = {}

                # Find left and top nodes if they exist
                left = node - 1 if j != 0 else None
                top = node - RESOLUTION if i != 0 else None

                # Case for one neighbor
                if isinstance(top, int) ^ isinstance(left, int):
                    # Identify left or top as the adjacent node
                    adj = top if isinstance(top, int) else left
                    a_val = df.iloc[adj // RESOLUTION, adj % RESOLUTION]

                    # Cluster if within tolerance of the adjacent node
                    if abs((n_val - a_val) / n_val) <= TOL:
                        cluster(node, adj)
                    # Don't cluster if not within tolerance
                    else:
                        nodes[node]["cluster"] = None

                # Case for two neighbors
                elif top and left:
                    # Fetch values of neighbors
                    t_val = df.iloc[i - 1, j]
                    l_val = df.iloc[i, j - 1]

                    # Cluster if within tolerance of top node
                    if abs((n_val - t_val) / n_val) <= TOL:
                        cluster(node, top)

                        # If also in tolerance of left node
                        if abs((n_val - l_val) / n_val) <= TOL:
                            cluster = nodes[node]["cluster"]
                            l_cluster = nodes[left]["cluster"]

                            cluster = cluster.union(l_cluster)
                            nodes[node]["cluster"] = cluster
                            nodes[top]["cluster"] = cluster
                            nodes[left]["cluster"] = cluster

                    # Cluster if only within tolerance of left node
                    elif abs((n_val - l_val) / n_val) <= TOL:
                        cluster(node, left)
                    else:
                        nodes[node]["cluster"] = None

                    pass



        pass


def main():
    make_graphs()


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