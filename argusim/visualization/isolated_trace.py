# This file is bascially a giant wrapper aroudn matplotlib's pyplot.
# reason for wrapping is these plots allow you to select, with your cursor, indiviusal traces and see their labels.
# This helps with debugging specific trails in a montecarlo analysis by knowing which rtrials is responsible for a strange plot

import matplotlib
from matplotlib import pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
import numpy as np
from matplotlib.widgets import TextBox

np.random.seed(1234)

PICKER_RADIUS = 5

matplotlib.rcParams.update({"axes.grid": True})
font = {"size": 20}
matplotlib.rc("font", **font)
plt.rc("legend", fontsize=10)  # using a size in points


class IsolatableTraceFigure:
    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.subplot_kwargs = {}
        self.shown_isolated_figures_before = False
        self.traces = []
        self.figure.canvas.mpl_connect("pick_event", self.__pick_handler)
        self.currently_focused_subplot_args = None
        self.called_subplot_before = False

    def plot(self, *args, **kwargs):
        kwargs.update({"picker": PICKER_RADIUS})

        # plot a single dot if there is only 1 datapoint
        n_datapoints = len(args[0])
        if n_datapoints == 1:
            kwargs.update({"marker": "o"})

        plt.figure(self.figure)
        ax = plt.gca() if isinstance(self.ax, np.ndarray) else self.ax
        (line,) = ax.plot(*args, **kwargs)
        self.traces.append((self.currently_focused_subplot_args, line))

        # save plot limits to be used when creating the isolated plots
        self.ax_plot_limits = []
        if isinstance(self.ax, np.ndarray):
            for ax in self.ax.flatten():
                self.ax_plot_limits.append((ax.get_xlim(), ax.get_ylim()))
        else:
            self.ax_plot_limits = [(self.ax.get_xlim(), self.ax.get_ylim())]

    def scatter(self, *args, **kwargs):
        kwargs.update({"picker": PICKER_RADIUS})

        plt.figure(self.figure)
        ax = plt.gca() if isinstance(self.ax, np.ndarray) else self.ax
        pathCollection = ax.scatter(*args, **kwargs)
        self.traces.append((self.currently_focused_subplot_args, pathCollection))

        # save plot limits to be used when creating the isolated plots
        self.ax_plot_limits = []
        if isinstance(self.ax, np.ndarray):
            for ax in self.ax.flatten():
                self.ax_plot_limits.append((ax.get_xlim(), ax.get_ylim()))
        else:
            self.ax_plot_limits = [(self.ax.get_xlim(), self.ax.get_ylim())]

    def subplot(self, *args, **kwargs):
        if not self.called_subplot_before:
            plt.close(self.figure)
            self.subplot_kwargs = {"nrows": args[0], "ncols": args[1]}
            self.figure, self.ax = plt.subplots(**self.subplot_kwargs)  # TODO dont create another plot
            self.figure.canvas.mpl_connect("pick_event", self.__pick_handler)
            self.called_subplot_before = True
        plt.figure(self.figure)
        self.currently_focused_subplot_args = args
        return plt.subplot(*args, **kwargs)

    def __pick_handler(self, event):
        if not self.shown_isolated_figures_before:
            self.isolated_figure, self.isolated_ax = plt.subplots(**self.subplot_kwargs)
            self.previous_coord = (0.0, 0.0)
            self.textbox_ax = plt.axes([0.50, 0.01, 0.04, 0.02])
            self.textbox = TextBox(ax=self.textbox_ax, initial="", label="Selected Trial: ")
            self.textbox.on_submit(self.__isolate_trial)

            self.shown_isolated_figures_before = True

        curr_coord = (event.mouseevent.xdata, event.mouseevent.ydata)
        if (
            event.mouseevent.button == 1
            and (isinstance(event.artist, Line2D) or isinstance(event.artist, PathCollection))
            and not curr_coord == self.previous_coord
        ):
            print("pick_handler")
            self.previous_coord = curr_coord
            artist = event.artist
            self.textbox.set_val(artist._label)  # triggers __isolate_trial
        self.previous_coord = curr_coord

    def __isolate_trial(self, trial_label):
        print("isolate_trial")

        matching_traces = []
        for subplot_args, trace in self.traces:
            if trace.get_label() == trial_label:
                matching_traces.append((subplot_args, trace))

        if matching_traces == []:
            print(f'No Trial is labeled: "{trial_label}"')
            return

        plt.figure(self.isolated_figure)
        for subplot_args, trace in matching_traces:
            if isinstance(self.ax, np.ndarray):
                plt.subplot(*subplot_args)
                plt.gca().clear()
                if isinstance(trace, Line2D):
                    plt.plot(trace.get_xdata(), trace.get_ydata(), color=trace.get_color())
                elif isinstance(trace, PathCollection):
                    plt.scatter(trace.get_offsets()[:, 0], trace.get_offsets()[:, 1], color=trace.get_facecolor())
            else:
                self.isolated_ax.cla()
                if isinstance(trace, Line2D):
                    self.isolated_ax.plot(trace.get_xdata(), trace.get_ydata(), color=trace.get_color())
                elif isinstance(trace, PathCollection):
                    self.isolated_ax.scatter(
                        trace.get_offsets()[:, 0], trace.get_offsets()[:, 1], color=trace.get_facecolor()
                    )
        self.copy_annotations()
        self.isolated_figure.canvas.draw()
        plt.show()

    def copy_annotations(self):
        plt.figure(self.isolated_figure)

        suptitle = self.figure._suptitle
        if suptitle is not None:
            plt.suptitle(self.figure._suptitle.get_text())

        isolated_axes = []
        axes = []
        if isinstance(self.ax, np.ndarray):
            isolated_axes = self.isolated_ax
            axes = self.ax
        else:
            isolated_axes = [self.isolated_ax]
            axes = [self.ax]

        for isolated_ax, ax, (x_plot_limits, y_plot_limits) in zip(isolated_axes, axes, self.ax_plot_limits):
            isolated_ax.set_title(ax.get_title())
            isolated_ax.set_xlabel(ax.get_xlabel())
            isolated_ax.set_ylabel(ax.get_ylabel())

    def set_size_inches(self, *args, **kwargs):
        self.figure.set_size_inches(*args, **kwargs)


class IsolatableTraceManager:
    def __init__(self):
        self.latest_itf = None
        self.rcParams = plt.rcParams

    def figure(self, itf=None, **kwargs):
        if itf is None:
            itf = IsolatableTraceFigure()
            self.latest_itf = itf
            return itf
        else:
            self.latest_itf = itf
            plt.figure(itf.figure, **kwargs)

    def subplot(self, *args, **kwargs):
        return self.latest_itf.subplot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.latest_itf.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self.latest_itf.scatter(*args, **kwargs)

    def hist(self, *args, **kwargs):
        plt.hist(*args, **kwargs)

    def title(self, *args, **kwargs):
        plt.title(*args, **kwargs)

    def figtext(self, *args, **kwargs):
        plt.figtext(*args, **kwargs)

    def suptitle(self, *args, **kwargs):
        plt.suptitle(*args, **kwargs)

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        plt.savefig(*args, **kwargs)

    def axvline(self, *args, **kwargs):
        return plt.axvline(*args, **kwargs)

    def axhline(self, *args, **kwargs):
        return plt.axhline(*args, **kwargs)

    def xlim(self, *args, **kwargs):
        return plt.xlim(*args, **kwargs)

    def ylim(self, *args, **kwargs):
        return plt.ylim(*args, **kwargs)

    def xlabel(self, *args, **kwargs):
        plt.xlabel(*args, **kwargs)

    def ylabel(self, *args, **kwargs):
        plt.ylabel(*args, **kwargs)

    def xticks(self, *args, **kwargs):
        plt.xticks(*args, **kwargs)

    def yticks(self, *args, **kwargs):
        plt.yticks(*args, **kwargs)

    def tick_params(self, *args, **kwargs):
        plt.tick_params(*args, **kwargs)

    def add_axes(self, *args, **kwargs):
        plt.add_axes(*args, **kwargs)

    def grid(self, *args, **kwargs):
        plt.grid(*args, **kwargs)

    def legend(self, *args, **kwargs):
        plt.legend(*args, **kwargs)

    def tight_layout(self, *args, **kwargs):
        plt.tight_layout(*args, **kwargs)

    def gcf(self, *args, **kwargs):
        return plt.gcf(*args, **kwargs)

    def gca(self, *args, **kwargs):
        return plt.gca(*args, **kwargs)

    def close(self, *args, **kwargs):
        plt.close(*args, **kwargs)


itm = IsolatableTraceManager()

# 'itm' replaces 'plt'

if __name__ == "__main__":
    fig1 = itm.figure()
    itm.scatter([13, 42, 43], [42, 3, 25], label="_abc")
    itm.scatter([11, 22, 13], [24, 31, 5], label="_def")
    plt.show()

    fig1 = itm.figure()
    fig2 = itm.figure()
    fig3 = itm.figure()
    time = np.linspace(0, 100, 10000)
    for trial in range(10):
        total = 0
        x = []
        y = []
        for t in time:
            r = np.random.rand() - 0.5
            x.append(r)
            y.append(total)
            total += r
        itm.figure(fig1)
        itm.plot(time, y, label=f"{trial}")

        itm.figure(fig2)
        itm.subplot(3, 1, 1)
        itm.plot(time, y, label=f"{trial}")
        itm.subplot(3, 1, 2)
        itm.plot(time, y, label=f"{trial}")
        itm.subplot(3, 1, 3)
        itm.plot(time, y, label=f"{trial}")

        itm.figure(fig3)
        itm.subplot(3, 1, 1)
        itm.plot(time, x, label=f"{trial}")
        itm.subplot(3, 1, 2)
        itm.plot(time, x, label=f"{trial}")
        itm.subplot(3, 1, 3)
        itm.plot(time, x, label=f"{trial}")

    itm.figure(fig1)
    plt.title("itf1 title")
    plt.xlabel("itf1 xlabel")
    plt.ylabel("itf1 ylabel")

    itm.figure(fig2)
    plt.suptitle("itf2 title")

    plt.subplot(3, 1, 1)
    plt.xlabel("itf2 1 xlabel")
    plt.ylabel("itf2 1 ylabel")

    plt.subplot(3, 1, 2)
    plt.xlabel("itf2 2 xlabel")
    plt.ylabel("itf2 2 ylabel")

    plt.subplot(3, 1, 3)
    plt.xlabel("itf2 3 xlabel")
    plt.ylabel("itf2 3 ylabel")

    itm.figure(fig3)
    plt.suptitle("itf3 title")

    plt.subplot(3, 1, 1)
    plt.xlabel("itf3 1 xlabel")
    plt.ylabel("itf3 1 ylabel")

    plt.subplot(3, 1, 2)
    plt.xlabel("itf3 2 xlabel")
    plt.ylabel("itf3 2 ylabel")

    plt.subplot(3, 1, 3)
    plt.xlabel("itf3 3 xlabel")
    plt.ylabel("itf3 3 ylabel")

    plt.show()

    # itf1 = IsolatableTraceFigure()
    # itf2 = IsolatableTraceFigure(nrows=3, ncols=1)
    # time = np.linspace(0, 100, 10000)
    # for trial in range(10):
    #     total = 0
    #     y = []
    #     for t in time:
    #         y.append(total)
    #         total += np.random.rand() - 0.5
    #     itf1.plot(time, y, label=f"{trial}")

    #     itf2.subplot(3, 1, 1)
    #     itf2.plot(time, y, label=f"{trial}")
    #     itf2.subplot(3, 1, 2)
    #     itf2.plot(time, y, label=f"{trial}")
    #     itf2.subplot(3, 1, 3)
    #     itf2.plot(time, y, label=f"{trial}")

    # plt.figure(itf1.figure)
    # plt.title("itf1 title")
    # plt.xlabel("itf1 xlabel")
    # plt.ylabel("itf1 ylabel")

    # plt.figure(itf2.figure)
    # plt.suptitle("itf2 title")

    # itf2.subplot(3, 1, 1)
    # plt.xlabel("itf2 1 xlabel")
    # plt.ylabel("itf2 1 ylabel")

    # itf2.subplot(3, 1, 2)
    # plt.xlabel("itf2 2 xlabel")
    # plt.ylabel("itf2 2 ylabel")

    # itf2.subplot(3, 1, 3)
    # plt.xlabel("itf2 3 xlabel")
    # plt.ylabel("itf2 3 ylabel")

    # plt.show()
