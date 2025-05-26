from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from pubmed_retina.exploration import find_mask_words


def plot_tsne_colors(
    tsne,
    colors,
    ax=None,
    x_lim=None,
    y_lim=None,
    plot_type=None,
    axis_on=False,
):
    """Plot t-SNE embedding with colors (by labels).

    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    axis_on : bool, default=False
        If True, axis is shown in plot.

    """
    if x_lim is not None:
        assert x_lim[0] < x_lim[1], "xlim values are in the wrong order."
    if y_lim is not None:
        assert y_lim[0] < y_lim[1], "ylim values are in the wrong order."

    assert plot_type in [
        None,
        "subplot_2",
        "subplot_3",
    ], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3']."

    if ax is None:
        fig, ax = plt.subplots()

    s_grey = 0.8
    s_color = 0.8
    alpha_grey = 0.6
    alpha_color = 0.7

    if plot_type == "subplot_2":
        s_grey = 0.2
        s_color = 0.2

    if plot_type == "subplot_3":
        s_grey = 0.1
        s_color = 0.1

    ax.scatter(
        tsne[:, 0][colors == "lightgrey"],
        tsne[:, 1][colors == "lightgrey"],
        s=s_grey,
        alpha=alpha_grey,
        c="lightgrey",
        marker=".",
        linewidths=0,
        ec="None",
        rasterized=True,
    )
    ax.scatter(
        tsne[:, 0][colors != "lightgrey"],
        tsne[:, 1][colors != "lightgrey"],
        s=s_color,
        alpha=alpha_color,
        c=colors[colors != "lightgrey"],
        marker=".",
        linewidths=0,
        ec="None",
        rasterized=True,
    )

    if plot_type == "subregion":
        ax.axis("scaled")
    else:
        ax.axis("equal")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if axis_on == False:
        ax.axis("off")


def find_cluster_center(tsne, colors, legend, subset=True, subset_size=500000, rs=42):
    """Find cluster centers.
    Finds coordinates of the highest density point of points from each label, using gaussian_kde.

    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.

    Returns
    -------
    center_cluster_coordinates_df : dataframe of shape (n_clusters, 2)
        Cluster center coordinates stored in two columns: "x" and "y".


    """

    words = list(legend.keys())
    unique_colors = np.array(list(legend.values()))

    if subset == True:
        np.random.seed(rs)
        assert tsne.shape[0] >= subset_size, "Subset size is smaller than dataset"
        index_subset = np.random.randint(0, tsne.shape[0], subset_size)
        tsne_subset = tsne[index_subset, :]
        colors_subset = colors[index_subset]

    else:
        tsne_subset = tsne
        colors_subset = colors

    # calculate cluster centers
    center_cluster_coordinates = []
    for i in range(len(words)):
        cluster = tsne_subset[colors_subset == unique_colors[i]]
        assert cluster.shape[0] > 0
        # center with kernel density
        kde = gaussian_kde(cluster.T)
        center_cluster_coordinates.append(cluster[kde(cluster.T).argmax()])

    center_cluster_coordinates = np.vstack(center_cluster_coordinates)

    center_cluster_coordinates_df = pd.DataFrame(
        center_cluster_coordinates, index=words, columns=["x", "y"]
    )

    return center_cluster_coordinates_df


def plot_label_tags(
    tsne,
    colors,
    legend,
    ax=None,
    x_lim=None,
    y_lim=None,
    middle_value=0,
    subset=True,
    subset_size=500000,
    rs=42,
    fontsize=7,
    capitalize=True,
    alpha_boxes=0.8,
):
    """Plots label tags and a line pointing to the embedding.
    The line from a label tag points to the location with higher points density of that specific label.


    Parameters
    ----------
    tsne: array-like of shape (n_points,2)
        t-SNE coordinates.
    colors : array-like of shape (n_points,)
        Color values for the colormap.
    legend : dict
        Legend label-color.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    middle_value : float, default=0
         The x value to decide which labels go to the left and which go to the right.
    subset : bool, default= True
         If True, a subset of the dataset is used for the cluster center calculations.
    subset_size : int, default=500000
        Size of the subset of the dataset used for the cluster center calculations.
    rs : int, default= 42
         Random seed.
    fontsize: int, default=7
         Fontsize for the labels.
    capitalize : bool, default = True
        If True, it will capitalize the labels.

    See Also
    --------
    find_cluster_center

    """

    if ax is None:
        fig, ax = plt.subplots()

    if x_lim is None:
        xmin, xmax, ymin, ymax = ax.axis()
        x_lim = (xmin, xmax)
    if y_lim is None:
        xmin, xmax, ymin, ymax = ax.axis()
        y_lim = (ymin, ymax)

    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"

    if "unlabeled" in legend.keys():
        legend.pop("unlabeled")

    # calculate cluster centers
    center_cluster_coordinates = find_cluster_center(
        tsne, colors, legend, subset, subset_size, rs
    )

    # sort by x
    center_cluster_coordinates_left = center_cluster_coordinates[
        center_cluster_coordinates.x < middle_value
    ].copy()
    center_cluster_coordinates_right = center_cluster_coordinates[
        center_cluster_coordinates.x >= middle_value
    ].copy()

    # sort by y
    center_cluster_coordinates_left.sort_values(by="y", inplace=True, ascending=False)
    center_cluster_coordinates_right.sort_values(by="y", inplace=True, ascending=False)

    sorted_labels_left = center_cluster_coordinates_left.index.tolist()
    sorted_labels_right = center_cluster_coordinates_right.index.tolist()

    sorted_colors_left = np.vectorize(legend.get)(sorted_labels_left)
    sorted_colors_right = np.vectorize(legend.get)(sorted_labels_right)

    if capitalize == True:
        sorted_labels_left = [elem.capitalize() for elem in sorted_labels_left]
        sorted_labels_right = [elem.capitalize() for elem in sorted_labels_right]

    # PLOT
    # left
    n_left = len(sorted_labels_left)
    x = x_lim[0] * np.ones(n_left)
    y = np.linspace(y_lim[1], y_lim[0], n_left)

    for i, colr in enumerate(sorted_colors_left):
        if any(
            [
                colr == "black",
                colr == "#0000A6",
                colr == "#5A0007",
                colr == "#4A3B53",
                colr == "#1B4400",
                colr == "#004D43",
                colr == "#013349",
                colr == "#000035",
                colr == "#300018",
                colr == "#001E09",
                colr == "#372101",
                colr == "#6508ba",
            ]
        ):
            # white colored letters
            ax.text(
                x[i],
                y[i],
                sorted_labels_left[i],
                c="lightgrey",
                fontsize=fontsize,
                ha="right",
                bbox=dict(
                    facecolor=colr,
                    edgecolor="None",
                    alpha=alpha_boxes,
                    boxstyle="square",
                    pad=0.05,
                ),
            )
            ax.plot(
                [x[i], center_cluster_coordinates_left.x[i]],
                [y[i], center_cluster_coordinates_left.y[i]],
                c=colr,
                linewidth=0.4,
                clip_on=False,
                alpha=alpha_boxes,
            )
        else:
            # black colored letters
            ax.text(
                x[i],
                y[i],
                sorted_labels_left[i],
                c="black",
                fontsize=fontsize,
                ha="right",
                bbox=dict(
                    facecolor=colr,
                    edgecolor="None",
                    alpha=alpha_boxes,
                    boxstyle="square",
                    pad=0.05,
                ),
            )
            ax.plot(
                [x[i], center_cluster_coordinates_left.x[i]],
                [y[i], center_cluster_coordinates_left.y[i]],
                c=colr,
                linewidth=0.4,
                clip_on=False,
                alpha=alpha_boxes,
            )

    # right
    n_right = len(sorted_labels_right)
    x = x_lim[1] * np.ones(n_right)
    y = np.linspace(y_lim[1], y_lim[0], n_right)

    for i, colr in enumerate(sorted_colors_right):
        # color blanco
        if any(
            [
                colr == "black",
                colr == "#0000A6",
                colr == "#5A0007",
                colr == "#4A3B53",
                colr == "#1B4400",
                colr == "#004D43",
                colr == "#013349",
                colr == "#000035",
                colr == "#300018",
                colr == "#001E09",
                colr == "#372101",
                colr == "#6508ba",
            ]
        ):
            ax.text(
                x[i],
                y[i],
                sorted_labels_right[i],
                c="lightgrey",
                fontsize=fontsize,
                ha="left",
                bbox=dict(
                    facecolor=colr,
                    edgecolor="None",
                    alpha=alpha_boxes,
                    boxstyle="square",
                    pad=0.05,
                ),
            )
            ax.plot(
                [x[i], center_cluster_coordinates_right.x[i]],
                [y[i], center_cluster_coordinates_right.y[i]],
                c=colr,
                linewidth=0.4,
                clip_on=False,
                alpha=alpha_boxes,
            )
        else:
            ax.text(
                x[i],
                y[i],
                sorted_labels_right[i],
                c="black",
                fontsize=fontsize,
                ha="left",
                bbox=dict(
                    facecolor=colr,
                    edgecolor="None",
                    alpha=alpha_boxes,
                    boxstyle="square",
                    pad=0.05,
                ),
            )
            ax.plot(
                [x[i], center_cluster_coordinates_right.x[i]],
                [y[i], center_cluster_coordinates_right.y[i]],
                c=colr,
                linewidth=0.4,
                clip_on=False,
                alpha=alpha_boxes,
            )


def plot_tsne_years(
    tsne,
    colors,
    x_lim,
    y_lim,
    ax=None,
    fontsize=7,
    plot_type=None,
    colorbar=True,
    colorbar_type=None,
    axis_on=False,
    rs=42,
    top_year="2021",
):
    """Plot t-SNE embedding with colors (by years).

    Parameters
    ----------
    tsne: array-like
        t-SNE coordinates.
    colors : array-like
        Color values for the colormap.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    fontsize : int, default=7
        Fontsize for the years in the colorbar.
    plot_type : {None, 'subplot', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    colorbar : bool, default=True
        If True, colorbar will be plotted.
    colorbar_type : {None, 'neuroscience'}, default=None
        Style of the colorbar.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    rs : int, default= 42
         Random seed for the reordering of points.

    """

    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order."
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order."
    assert type(top_year) == str, "top_year should be a string."

    assert plot_type in [
        None,
        "subplot",
        "subregion",
        "test",
    ], "Not valid `plot_type` value. Choose from [None, 'subplot', 'subregion', 'test']."
    assert colorbar_type in [
        None,
        "neuroscience",
    ], "Not valid `colorbar_type` value. Choose from [None, 'neuroscience']."

    if ax is None:
        fig, ax = plt.subplots()

    s_color = 0.5
    alpha_color = 0.2

    if plot_type == "subplot":
        s_color = 0.2
        alpha_color = 0.2

    if plot_type == "subregion":
        s_color = 0.5
        alpha_color = 0.7

    if plot_type == "test":
        s_color = 2
        alpha_color = 0.7

    np.random.seed(rs)
    reorder = np.random.permutation(tsne.shape[0])
    ax.scatter(
        tsne[reorder][:, 0],
        tsne[reorder][:, 1],
        s=s_color,
        c=colors[reorder],
        cmap="plasma",
        alpha=alpha_color,
        marker=".",
        linewidths=0,
        rasterized=True,
    )

    if plot_type == "subregion":
        ax.axis("scaled")
    else:
        ax.axis("equal")

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    if axis_on == False:
        ax.axis("off")

    if colorbar == True:
        if colorbar_type == "neuroscience":
            heatmap = ax.scatter([], [], c=[], cmap="plasma")
            cbar = plt.colorbar(
                heatmap,
                ax=ax,
                shrink=0.1,
                location="left",
                anchor=(0, 0),
                panchor=(0, 0),
                pad=-0.3,
                aspect=10,
            )

        else:
            heatmap = ax.scatter([], [], c=[], cmap="plasma")
            cbar = plt.colorbar(
                heatmap,
                ax=ax,
                shrink=0.1,
                anchor=(0.5, 0),
                panchor=(0, 0.5),
                pad=-0.13,
                aspect=10,
            )
            # anchor second coordinate controls y-position and pad controls x-position

        cbar.set_alpha(1)
        cbar.ax.get_yaxis().set_ticks([0, 1])
        cbar.ax.get_yaxis().set_ticklabels(["1970", top_year])
        cbar.ax.tick_params(labelsize=fontsize)


def plot_tsne_mask(mask, tsne, x_lim, y_lim, ax=None, plot_type=None, axis_on=False):
    """Plots t-SNE embedding with points belonging to a mask highlighted.
    It plots all points in grey, and papers that have that specific word/phrase in their abstract in black.
    If more than one word is given, each of them will be plotted using colors from tab10 color palette.
    Take into account that if this happens, points will be plotted on top of each other instead of shuffled, so the amount of papers may be missleading.

    Parameters
    ----------
    all_abstracts : pandas dataframe of str
        All texts (in this case abstracts).
    words : str or list of str
        Word/phrase or list with many words/phrases to be queried.
    tsne: array-like
        t-SNE coordinates.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    title_on : bool, default=False
        If True, adds the word being queried as title to the figure.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    verbose : bool, default=True
        If True, prints the number of papers with that certain word and its variations in it .

    See Also
    --------
    exploration.find_mask_words

    """

    assert x_lim[0] < x_lim[1], "xlim values are in the wrong order"
    assert y_lim[0] < y_lim[1], "ylim values are in the wrong order"

    assert plot_type in [
        None,
        "subplot_2",
        "subplot_3",
        "subplot_3_grey",
        "subregion",
        "test",
    ], "Not valid `plot_type` value. Choose from [None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test']."

    # s_grey = 0.1
    # s_color = 0.5
    # alpha_grey = 0.2
    # alpha_color = 0.5

    s_grey = 0.5
    s_color = 0.5
    alpha_grey = 0.6
    alpha_color = 0.7

    if plot_type == "subplot_2":
        s_grey = 0.2
        s_color = 0.2

    if plot_type == "subplot_3":
        s_grey = 0.1
        s_color = 0.1

    if plot_type == "subplot_3_grey":
        s_grey = 0.05
        alpha_grey = 0.01
        s_color = 0.2
        alpha_color = 0.2  # 0.5

    if plot_type == "subregion":
        s_grey = 1
        s_color = 1
        alpha_grey = 0.6
        alpha_color = 0.7

    if plot_type == "test":
        s_grey = 2
        s_color = 2
        alpha_grey = 0.6
        alpha_color = 0.7

    ax_passed = True
    if ax is None:
        ax_passed = False
        fig, ax = plt.subplots()

    subregion = tsne[mask]
    ax.scatter(
        tsne[:, 0],
        tsne[:, 1],
        c="lightgrey",
        s=s_grey,
        alpha=alpha_grey,
        edgecolors="None",
        rasterized=True,
    )  # ,linewidths=0)
    ax.scatter(
        subregion[:, 0],
        subregion[:, 1],
        s=s_color,
        c="black",
        alpha=alpha_color,
        marker=".",
        edgecolors="None",
        rasterized=True,
    )  # linewidths=0)

    ax.axis("equal")
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    if axis_on == False:
        ax.axis("off")

    return None if ax_passed else (fig, ax)


def plot_tsne_word(
    all_abstracts,
    word,
    tsne,
    x_lim,
    y_lim,
    ax=None,
    plot_type=None,
    title_on=True,
    verbose=True,
    save_fig=False,
    saving_path=None,
):
    """Plots t-SNE embedding with points having one given word in their abstract highlighted.
    It plots all points in grey, and papers that have that specific word/phrase in their abstract in black.
    If more than one word is given, each of them will be plotted using colors from tab10 color palette.
    Take into account that if this happens, points will be plotted on top of each other instead of shuffled, so the amount of papers may be missleading.


    Parameters
    ----------
    all_abstracts : pandas dataframe of str
        All texts (in this case abstracts).
    words : str or list of str
        Word/phrase or list with many words/phrases to be queried.
    tsne: array-like
        t-SNE coordinates.
    x_lim : tuple (left, right)
        Limits of the x-axis.
    y_lim : tuple (bottom, top)
        Limits of the y-axis.
    ax : axes, optional
        Axes where to draw the figure. If ax=None, axes will be created.
    plot_type : {None, 'subplot_2', 'subplot_3', 'subplot_3_grey', 'subregion', 'test'}, default=None
        Style of the plot, modifies dotsize and alpha.
    title_on : bool, default=False
        If True, adds the word being queried as title to the figure.
    axis_on : bool, default=False
        If True, axis is shown in plot.
    verbose : bool, default=True
        If True, prints the number of papers with that certain word and its variations in it .

    See Also
    --------
    exploration.find_mask_words

    """
    assert (save_fig == True) & (saving_path is not None), "Saving path is missing"

    if type(word) is str:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        mask = find_mask_words(all_abstracts, word, verbose=verbose)
        plot_tsne_mask(
            mask, tsne, x_lim, y_lim, ax=ax, plot_type=plot_type, axis_on=False
        )
        if title_on:
            ax.set_title(f"'{word}'")
        if save_fig:
            fig.savefig(saving_path / f"tsne_mask_{word}.png")

    if type(word) is list:
        for wrd in word:
            fig, ax = plt.subplots(figsize=(4, 4))
            mask = find_mask_words(all_abstracts, wrd, verbose=verbose)
            plot_tsne_mask(
                mask, tsne, x_lim, y_lim, ax=ax, plot_type=plot_type, axis_on=False
            )
            if title_on:
                ax.set_title(f"'{wrd}'")
            if save_fig:
                fig.savefig(saving_path / f"tsne_mask_{wrd}.png")
