from isolated_trace import itm
import numpy as np
import os

# Enable grid lines by default
itm.rcParams["axes.grid"] = True


def plotWrapper(x, y, sublpotRows, subplotCols, positionInSubplot, label, title, xlabel, ylabel):
    itm.subplot(sublpotRows, subplotCols, positionInSubplot)
    itm.named_plot(label, x, y)
    itm.title(title)
    itm.xlabel(xlabel)
    itm.ylabel(ylabel)
    if not label.empty():
        itm.legend()

    itm.grid(True)


def plotUncertaintyEnvelope(time, x, stddev_x, sigmas):
    # 1-sigma envelope
    itm.fill_between(time, x - stddev_x, x + stddev_x, color="green", alpha=0.4, label="1-sigma")

    if sigmas != 1.0:  # multi-sigma envelope. purposely comapring to float
        # lower envelope
        itm.fill_between(time, x - sigmas * stddev_x, x - stddev_x, color="red", alpha=0.4, label=f"{sigmas:.3f}-sigma")

        # upper envelope
        itm.fill_between(time, x + stddev_x, x + sigmas * stddev_x, color="green", alpha=0.4)
    itm.legend()


# TOOD optional scale factor?
def triPlot(time, series, color=None, seriesLabel=None, **kwargs):
    if color is not None:
        kwargs["color"] = color
    if seriesLabel is not None:
        kwargs["label"] = seriesLabel

    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 0.3

    itm.subplot(3, 1, 1)
    itm.plot(time, series[0], **kwargs)
    itm.legend()

    itm.subplot(3, 1, 2)
    itm.plot(time, series[1], **kwargs)
    itm.legend()

    itm.subplot(3, 1, 3)
    itm.plot(time, series[2], **kwargs)
    itm.legend()

    annotateTriPlot()


def annotatePlot(title=None, xlabel=None, ylabel=None, seriesLabel=None):
    if title is not None:
        itm.title(title)
    if xlabel is not None:
        itm.xlabel(xlabel)
    if ylabel is not None:
        itm.ylabel(ylabel)
    if seriesLabel is not None:
        itm.legend({seriesLabel})
    itm.grid(True, linewidth=0.4)


def plot_orientations(time, orientations, series_label, title, labelDegrees, color):
    euler_angles = []
    for orientation in orientations:
        euler_angle = intrinsic_zyx_decomposition(orientation)
        if labelDegrees:
            euler_angle = RAD_2_DEG(euler_angle)
        euler_angles.append(euler_angle)
    triPlot(time, euler_angles, color=color)


def annotateTriPlot(y_units=None, title=None, ylabels=["x", "y", "z"], seriesLabel=None):
    if title is not None:
        itm.suptitle(title)

    for i in range(3):
        itm.subplot(3, 1, i + 1)
        full_y_label = ylabels[i]
        if y_units is not None:
            units_suffix = " [" + y_units + "]"
            full_y_label += units_suffix
        annotatePlot(xlabel="time [s]", ylabel=full_y_label, seriesLabel=seriesLabel)


# TOOD optional scale factor?
def triBounds(time, stddev_series):
    itm.subplot(3, 1, 1)
    plotUncertaintyEnvelope(time, np.zeros(len(time)), stddev_series.x, 3)

    itm.subplot(3, 1, 2)
    plotUncertaintyEnvelope(time, np.zeros(len(time)), stddev_series.y, 3)

    itm.subplot(3, 1, 3)
    plotUncertaintyEnvelope(time, np.zeros(len(time)), stddev_series.z, 3)


def comparePlots(time, true_series, estimated_series, y_units, title, ylabels):
    triPlot(time, true_series)
    annotateTriPlot(y_units, title, ylabels)
    triPlot(time, estimated_series)
    annotateTriPlot(y_units, title, ylabels)

    for i in range(3):
        itm.subplot(3, 1, i + 1)
        itm.legend({"True", "estimated"})


def showError(time, trueSeries, estimatedSeries, color):
    triPlot(time, trueSeries - estimatedSeries, color)


def draw_vertical_line(color, x, label):
    return itm.axvline(x, color=color, label=rf"{label}: {x:.4g}", linestyle="dashed", linewidth=2)


def draw_horizontal_line(color, y, label=None, linestyle="dashed", linewidth=2):
    return itm.axhline(y, color=color, label=label, linestyle=linestyle, linewidth=linewidth)


def plot_hist(figure, data, xlabel, title, density=False):
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
    itm.figure(figure)

    # Adjust figure size to match maximized window
    figure.set_size_inches(13, 8)

    # Save the figure
    itm.savefig(os.path.join(folder, filename), bbox_inches="tight")
    if close_after_saving:
        itm.close()
