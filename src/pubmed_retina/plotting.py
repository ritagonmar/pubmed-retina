from matplotlib import pyplot as plt


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
    ], "Not valid `plot_type` value. Choose from [None, 'subplot_2']."

    if ax is None:
        fig, ax = plt.subplots()

    s_grey = 0.5
    s_color = 0.5
    alpha_grey = 0.6
    alpha_color = 0.7

    if plot_type == "subplot_2":
        s_grey = 0.2
        s_color = 0.2

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