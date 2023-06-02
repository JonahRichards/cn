import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob

import matplotlib.animation as animation

'''
# Path to the directory containing the PNG images
image_directory = 't1g\\'

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
animation_object = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=1000)

# Set the number of frames per second (optional)
frames_per_second = 15
animation_object.save('animation2.gif', fps=frames_per_second, writer='ffmpeg')

'''


tol = 0.99

# defining edges
def get_edges(df):
    edges = []
    for i in range(0, 7):
        for j in range(0, 7):
            name = i * 8 + j

            node = df.iloc[i].iloc[j]
            right = df.iloc[i + 1].iloc[j]
            down = df.iloc[i].iloc[j + 1]

            if math.fabs((node - right) / node) >= tol:
                edges.append((name, name + 1))
            if math.fabs((node - down) / node) >= tol:
                edges.append((name, name + 8))
    return edges


'''data = pd.read_csv("test.csv", usecols=range(20000, 20017))
extra = data.head(6)
data = pd.concat([data, extra])'''

for i in range(100):
    data = pd.read_csv(f"t1\\{i}.csv")
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='magma')

    # Hide the axes
    ax.axis('off')

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Save the plot to a directory
    plt.savefig(f't1h\\{i}.png')

    edges = get_edges(data)

    fig, ax = plt.subplots()
    graph = nx.Graph(edges)
    [graph.add_node(i) for i in range(256)]

    pos = {k: (k % 16, k // -16) for k in range(256)}
    nx.draw(graph, pos, node_size=50, node_color='red', edge_color='black', width=4)

    plt.savefig(f't1g\\{i}g.png')

    pass


for i in range(20):
    data = pd.read_csv(f"t1\\{i}.csv")
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data, cmap='magma')

    # Hide the axes
    ax.axis('off')

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Save the plot to a directory
    plt.savefig(f't2h\\{i}.png')

    edges = get_edges(data)

    fig, ax = plt.subplots()
    graph = nx.Graph(edges)
    [graph.add_node(i) for i in range(64)]

    pos = {k: (k % 8, k // -8) for k in range(64)}
    nx.draw(graph, pos, node_size=50, node_color='red', edge_color='black', width=4)

    plt.savefig(f't2g\\{i}g.png')

    pass


