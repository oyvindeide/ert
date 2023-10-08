import numpy as np
from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QTreeWidget, QTreeWidgetItem, QMainWindow, \
    QAbstractItemView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functools import wraps

from scipy.stats import gaussian_kde


def plot_changed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self.axes.legend()
        self.draw()
    return wrapper


class ResponsePlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.subplots(nrows=1, ncols=1)
        self.secondary_axes = self.axes.twinx()
        self.active_plots = {}
        self.secondary_plots = {}

    @plot_changed
    def plot(self, plot_id, data):
        line = data.plot.scatter(ax=self.axes, label=plot_id)
        self.active_plots[plot_id] = line

    @plot_changed
    def plot_histogram(self, plot_id, data):
        _, _, bars = data.plot.hist(ax=self.axes, label=plot_id)
        self.active_plots[plot_id] = bars

    @plot_changed
    def plot_kde(self, plot_id, data):
        sample_range = data.max() - data.min()
        indexes = np.linspace(
            data.min() - 0.5 * sample_range, data.max() + 0.5 * sample_range, 1000
        )

        gkde = gaussian_kde(data.values)
        evaluated_gkde = gkde.evaluate(indexes)
        lines = self.secondary_axes.plot(
            indexes,
            evaluated_gkde,
            color=self.active_plots[plot_id][0].get_facecolor()
        )
        self.secondary_plots[plot_id] = lines

    @plot_changed
    def remove_plot(self, plot_id):
        plot = self.active_plots.pop(plot_id)
        try:
            [b.remove() for b in plot]
        except TypeError:
            plot.remove()


class PlotWindow(QMainWindow):
    def __init__(self, storage, parent=None):
        QMainWindow.__init__(self, parent)

        self.activateWindow()
        layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        tree = QTreeWidget()
        tree.setSelectionMode(QAbstractItemView.MultiSelection)
        tree.setColumnCount(3)
        self.tree = tree
        self.ens = storage.get_ensemble_by_name("default_smoother")

        self.responses = self.ens.experiment.response_configuration
        self.parameters = self.ens.experiment.parameter_configuration

        for data_type, config in zip(["response", "parameter"], [self.responses, self.parameters]):
            parent_item = QTreeWidgetItem(tree)
            parent_item.setText(0, data_type)
            for response in config.values():
                sub_keys = response.plot_settings.keys
                item = QTreeWidgetItem(parent_item)
                parent_item.addChild(item)
                item.setText(1, response.name)
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                for key in sub_keys:
                    child_item = QTreeWidgetItem(item)
                    child_item.setText(2, str(key))
                    item.addChild(child_item)

        self.fm_pyplot = ResponsePlot()
        layout.addWidget(self.tree)
        layout.addWidget(self.fm_pyplot, 1)
        self.setWindowTitle("QHBoxLayout Example")
        self.tree.itemSelectionChanged.connect(
            self.change_plot
        )
        self.show()

    def change_plot(self):
        current_active = []
        for selected in self.tree.selectedItems():
            if selected is None:
                continue
            plot_type = selected.parent().parent().text(0)
            sub_key = selected.text(2)
            main_key = selected.parent().text(1)
            plot_id = f"{main_key}:{sub_key}"
            current_active.append(plot_id)
            if plot_id in self.fm_pyplot.active_plots:
                continue
            if plot_type == "response":
                ensemble = self.ens.load_response(main_key, tuple(range(self.ens.ensemble_size)))
                self.fm_pyplot.plot(plot_id, ensemble.sel({self.responses[main_key].plot_settings.name:self.responses[main_key].plot_settings.key_type(sub_key)})["values"])
            else:
                ensemble = self.ens.load_parameters(main_key)
                self.fm_pyplot.plot_histogram(plot_id, ensemble.sel({self.parameters[main_key].plot_settings.name:self.parameters[main_key].plot_settings.key_type(sub_key)}))
        deactivated = [id for id in self.fm_pyplot.active_plots if id not in current_active]
        for d in deactivated:
            self.fm_pyplot.remove_plot(d)
