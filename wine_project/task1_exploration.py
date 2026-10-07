"""
task1_exploration.py

Task 1 - Data exploration of the red wine quality dataset.

Task 1.1  correlation_matrix()  
    - Pearson correlations of the physicochemical properties, 
    drawn as a matplotlib heatmap with dendrograms at the sides,
    saved to task1_heatmap.png
Task 1.2  plot_tsne()           
    - tSNE transform of the data, plotted with colour and labelling 
    that make the quality structure readable, saved to task1_tsne.png
    
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import winequality

HEATMAP_FILE = "task1_heatmap.png"
TSNE_FILE = "task1_tsne.png"

# Quality is an ordinal score from 3 to 8.  Most wines score 5 or 6, so we
# group the scores into three bands and highlight the rare low and high wines:
# low = 3-4, medium = 5-6, high = 7-8.
LOW_MAX = 4       # scores up to here are "low"
MEDIUM_MAX = 6    # scores up to here (and above LOW_MAX) are "medium"; rest "high"


def get_column_names(data_file=winequality.DATA_FILE):
    """
    Read just the header row of a data file and return the column names.

    data_file - path to the CSV file; defaults to winequality.DATA_FILE

    The names are returned in the same order as the columns in the
    array returned by winequality.load_data().
    """
    header = pd.read_csv(data_file, sep=winequality.SEP, nrows=0)
    return list(header.columns)


def correlation_matrix(data):
    """
    Calculate Pearson correlations between all properties and quality.

    data - numpy array of values, as returned by winequality.load_data(),
           with one row per wine and one column per property (quality last)

    Returns the correlation matrix as a 2D numpy array, with one row and
    one column per property, in the same order as data's columns.
    """
    # rowvar=False: each column is a variable, each row an observation.
    return np.corrcoef(data, rowvar=False)

def cluster_order(correlation_matrix):
    """
    Cluster properties that behave alike, using the correlation matrix.

    correlation_matrix - 2D numpy array, as returned by correlation_matrix()

    Returns (order, link):
    order - list of row/column indices, in clustered order
    link  - the linkage matrix, for drawing a dendrogram with the same order
    """
    # 1 - correlation as a distance: identical properties have distance 0.
    # Assumption: strongly negative correlations (r near -1) give a distance
    # near 2, so properties that move in opposite directions are placed far
    # apart, not clustered together.
    distance = 1 - correlation_matrix
    np.fill_diagonal(distance, 0)
    # squareform turns the square distance matrix into the condensed
    # form that linkage() expects.
    condensed = squareform(distance, checks=False)
    link = linkage(condensed, method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    return order, link


def annotate_cells(ax, matrix):
    """
    Write each correlation value in its cell on the heatmap ax.

    Strong correlations sit on dark squares, so values with |r| > 0.6 are
    written in white to stay readable; the rest are black.  A tiny negative
    that would round to "-0.00" is shown as "0.00".
    """
    n = matrix.shape[0]
    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            text = f"{value:.2f}"
            if text == "-0.00":
                text = "0.00"
            colour = "white" if abs(value) > 0.6 else "black"
            ax.text(col, row, text, ha="center", va="center",
                    fontsize=6, color=colour)


def plot_heatmap(correlation_matrix, labels, output_file=HEATMAP_FILE):
    """
    Plot correlation matrix as a heatmap with a dendrogram
    on each side and save to output_file.

    correlation_matrix - 2D numpy array, as returned by correlation_matrix()
    labels             - property names, one per row/column, in the same
                          order as correlation_matrix (e.g. from get_column_names())
    output_file        - path to save the picture to
    """
    order, link = cluster_order(correlation_matrix)
    ordered_matrix = correlation_matrix[order][:, order]
    ordered_labels = [labels[i] for i in order]
    n = len(ordered_labels)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Correlation of wine properties")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 4), height_ratios=(1, 4),
                             wspace=0.02, hspace=0.02)

    # Top-left corner is otherwise empty; put a small horizontal colour
    # bar there, fixed to [-1, 1] so it can be drawn before the heatmap.
    ax_cbar = fig.add_subplot(grid[0, 0])
    ax_cbar.set_box_aspect(0.15)  # make the colour bar a thin strip, not a square
    norm = plt.Normalize(vmin=-1, vmax=1)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                 cax=ax_cbar, orientation="horizontal",
                 label="Pearson correlation")
    ax_cbar.xaxis.set_ticks_position("top")
    ax_cbar.tick_params(labelsize=6)

    # Top dendrogram (columns).
    ax_top = fig.add_subplot(grid[0, 1])
    dendrogram(link, ax=ax_top, no_labels=True)
    ax_top.axis("off")

    # Left dendrogram (rows).  scipy draws leaf 0 at the bottom, but
    # imshow draws row 0 at the top, so flip the axis to line them up.
    ax_left = fig.add_subplot(grid[1, 0])
    dendrogram(link, ax=ax_left, orientation="left", no_labels=True)
    ax_left.axis("off")
    ax_left.invert_yaxis()

    # Main heatmap, with a colour scale fixed to [-1, 1] so positive and
    # negative correlations are always easy to tell apart.  aspect="auto"
    # keeps the heatmap lined up with both dendrograms.
    ax_heat = fig.add_subplot(grid[1, 1])
    image = ax_heat.imshow(ordered_matrix, cmap="RdBu_r", vmin=-1, vmax=1,
                            aspect="auto")

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(ordered_labels, rotation=45, ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.yaxis.tick_right()
    ax_heat.set_yticklabels(ordered_labels)

    # Thin white lines between the squares, drawn on the cell boundaries.
    ax_heat.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1)
    ax_heat.tick_params(which="minor", length=0)

    annotate_cells(ax_heat, ordered_matrix)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

def plot_tsne(data_file=winequality.DATA_FILE, perplexity=60,
              output_file=TSNE_FILE):
    """
    Transform the data with tSNE and plot the result.

    perplexity - sklearn.manifold.TSNE parameter; the default is 30, but 60
                  may separate this dataset better

    Saves the plot to output_file.
    """
    data = winequality.load_data(data_file)
    properties = data[:, :-1]  # the 9 chemistry columns
    quality = data[:, -1]      # used only for colouring the points

    # tSNE is distance-based, so put all properties on the same scale.
    scaled = StandardScaler().fit_transform(properties)
    embedding = TSNE(perplexity=perplexity, random_state=0).fit_transform(scaled)

    # Split the wines into the three quality groups.
    low = quality <= LOW_MAX
    medium = (quality > LOW_MAX) & (quality <= MEDIUM_MAX)
    high = quality > MEDIUM_MAX

    fig, ax = plt.subplots(figsize=(8, 8))
    # Draw the common medium wines first as a light grey backdrop, then the
    # rare low and high wines on top in strong colours with distinct shapes.
    ax.scatter(embedding[medium, 0], embedding[medium, 1], s=10, alpha=0.4,
               color="lightgrey", label=f"medium (5-6): {int(medium.sum())}")
    ax.scatter(embedding[low, 0], embedding[low, 1], s=10, alpha=0.7,
               color="crimson", marker="v", label=f"low (3-4): {int(low.sum())}")
    ax.scatter(embedding[high, 0], embedding[high, 1], s=10, alpha=0.7,
               color="royalblue", marker="^", label=f"high (7-8): {int(high.sum())}")

    ax.set_xlabel("tSNE 1")
    ax.set_ylabel("tSNE 2")
    ax.set_title(f"tSNE of wine properties by quality group "
                 f"(perplexity={perplexity})")
    ax.legend()

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = winequality.load_data()
    labels = get_column_names()
    corr = correlation_matrix(data)
    plot_heatmap(corr, labels)
    plot_tsne()
