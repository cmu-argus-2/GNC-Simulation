from argusim.visualization.isolated_trace import itm
import numpy as np
import os

# Enable grid lines by default
itm.rcParams["axes.grid"] = True


def plotWrapper(x, y, sublpotRows, subplotCols, positionInSubplot, label, title, xlabel, ylabel):
    """All-in-1 utiltiy function for creating a single subplot in a grid of plots

    Args:
        x (Nx1 list): x data
        y (Nx1 list): y data
        sublpotRows (int): same as the arg passed into plt.subplot()
        subplotCols (int): same as the arg passed into plt.subplot()
        positionInSubplot (int): same as the arg passed into plt.subplot()
        label (str): label to display in legend
        title (str): title
        xlabel (str): xlabel
        ylabel (str): ylabel
    """
    itm.subplot(sublpotRows, subplotCols, positionInSubplot)
    itm.named_plot(label, x, y)
    itm.title(title)
    itm.xlabel(xlabel)
    itm.ylabel(ylabel)
    if not label.empty():
        itm.legend()

    itm.grid(True)


def multiPlot(time, multiSeries, seriesLabel=None, **kwargs):
    """Plots Multiple time series stacked vertically

    Args:
        time (Nx1 list): timestamps
        multiSeries (M x N list): M time series each with N datapoints
        seriesLabel (str, optional): The label for legend. Defaults to None.
    """
    if seriesLabel is not None:
        kwargs["label"] = seriesLabel

    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 0.3

    N = len(multiSeries)
    for i, series in enumerate(multiSeries):
        itm.subplot(N, 1, i + 1)
        itm.plot(time, series, **kwargs)
        itm.legend()

    annotateMultiPlot()


def annotatePlot(title=None, xlabel=None, ylabel=None, seriesLabel=None):
    """Label a single plot

    Args:
        title (str, optional): Defaults to None.
        xlabel (str, optional): Defaults to None.
        ylabel (str, optional): Defaults to None.
        seriesLabel (str, optional): The label to shows in the legend. Defaults to None.
    """
    if title is not None:
        itm.title(title)
    if xlabel is not None:
        itm.xlabel(xlabel)
    if ylabel is not None:
        itm.ylabel(ylabel)
    if seriesLabel is not None:
        itm.legend({seriesLabel})
    itm.grid(True, linewidth=0.4)


def annotateTriPlot(y_units=None, title=None, ylabels=["x", "y", "z"], seriesLabel=None):
    """Like annotatePlot but specifically for figures with 3 plots stacked verically.

    Args:
        y_units (list[str], optional): _description_. Defaults to None.
        title (str, optional): _description_. Defaults to None.
        ylabels (list[str], optional): _description_. Defaults to ["x", "y", "z"].
        seriesLabel (list[str], optional): _description_. Defaults to None.
    """
    if title is not None:
        itm.suptitle(title)

    for i in range(3):
        itm.subplot(3, 1, i + 1)
        full_y_label = ylabels[i]
        if y_units is not None:
            units_suffix = " [" + y_units + "]"
            full_y_label += units_suffix
        annotatePlot(xlabel="time [s]", ylabel=full_y_label, seriesLabel=seriesLabel)


def annotateMultiPlot(y_units=None, title=None, ylabels=None, seriesLabel=None):
    """Like annotateTriPlot() but nothing is limitsed to 2 options.

    Args:
        y_units (_type_, optional): _description_. Defaults to None.
        title (_type_, optional): _description_. Defaults to None.
        ylabels (_type_, optional): _description_. Defaults to None.
        seriesLabel (_type_, optional): _description_. Defaults to None.
    """
    if title is not None:
        itm.suptitle(title)

    if ylabels is not None:
        N = len(ylabels)
        for i in range(N):
            itm.subplot(N, 1, i + 1)
            full_y_label = ylabels[i]
            if y_units is not None:
                units_suffix = " [" + y_units + "]"
                full_y_label += units_suffix
            annotatePlot(xlabel="time [s]", ylabel=full_y_label, seriesLabel=seriesLabel)


def draw_vertical_line(color, x, label):
    return itm.axvline(x, color=color, label=rf"{label}: {x:.4g}", linestyle="dashed", linewidth=2)


def draw_horizontal_line(color, y, label=None, linestyle="dashed", linewidth=2):
    return itm.axhline(y, color=color, label=label, linestyle=linestyle, linewidth=linewidth)


def plot_hist(figure, data, xlabel, title, density=False):
    """Convenience function for creating and annoting a histogram in 1 go. Also automatically computes and plots the mean .median, 1 and 3 sigma bounds

    Args:
        figure (plt.figure): figure onwhic to plot the histogram
        data (Nx1 list): data
        xlabel (_type_):
        title (_type_):
        density (bool, optional): Whether to make a pdf. Defaults to False.
    """
    itm.figure(figure)
    itm.grid(True, zorder=1)  # render grid behind the histogram
    itm.hist(
        data,
        density=density,
        color="lightgreen",
        edgecolor="black",
        zorder=2,
    )
    if title is not None:
        itm.title(title)
    if xlabel is not None:
        itm.xlabel(xlabel)
    if density:
        itm.ylabel("pdf")
    else:
        itm.ylabel("counts")

    median = np.median(data)
    mean = np.mean(data)
    sigma = np.std(data)

    itm.legend(
        handles=[
            draw_vertical_line("r", mean - 3 * sigma, "$\mu-3\sigma$"),
            draw_vertical_line("m", mean - sigma, "$\mu-\sigma$"),
            draw_vertical_line("b", median, "median"),
            draw_vertical_line("g", mean, "$\mu$"),
            draw_vertical_line("m", mean + sigma, "$\mu+\sigma$"),
            draw_vertical_line("r", mean + 3 * sigma, "$\mu+3\sigma$"),
        ]
    )


def save_figure(figure, folder, filename, close_after_saving=True):
    """Saves an image of the figure to the filesystem

    Args:
        figure (plt.figure): _description_
        folder (str): directory in which to save the image
        filename (str): the img name ot save
        close_after_saving (bool, optional): Whether to close the figure with plt.close(). Defaults to True.
    """
    itm.figure(figure)

    # Adjust figure size to match maximized window
    figure.set_size_inches(13, 8)

    # Save the figure
    itm.savefig(os.path.join(folder, filename), bbox_inches="tight")
    if close_after_saving:
        itm.close()
