from PyQt5.Qt import Qt
from PyQt5.QtGui import QWindow
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QTreeWidget, QTreeWidgetItem, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xarray as xr

from ert.storage import open_storage

import sys


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.data = xr.DataArray()
        self.axes = self.fig.subplots(nrows=1, ncols=1)

    def plot(self, data):
        self.fig.axes.clear()
        self.data = data
        self.data.plot.scatter(ax=self.axes)
        self.draw()


class PlotWindow(QMainWindow):
    def __init__(self, storage, parent=None):
        QMainWindow.__init__(self, parent)

        self.activateWindow()
        layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        tree = QTreeWidget()
        tree.setColumnCount(2)
        self.tree = tree
        self.ens = storage.get_ensemble_by_name("default_smoother")

        responses = self.ens.experiment.response_configuration
        self.responses = responses
        for response in responses.values():
            sub_keys = response.plot_settings.keys
            item = QTreeWidgetItem(tree)
            item.setText(0, response.name)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            for key in sub_keys:
                child_item = QTreeWidgetItem(item)
                child_item.setText(1, str(key))
                item.addChild(child_item)

        self.fm_pyplot = PlotCanvas()
        layout.addWidget(self.tree)
        layout.addWidget(self.fm_pyplot, 1)
        self.setWindowTitle("QHBoxLayout Example")
        self.tree.itemSelectionChanged.connect(
            self.change_plot
        )
        self.show()

    def change_plot(self):
        sub_key = self.tree.selectedItems()[0].text(1)
        main_key = self.tree.selectedItems()[0].parent().text(0)
        ensemble = self.ens.load_response(main_key, tuple(range(self.ens.ensemble_size)))
        self.fm_pyplot.plot(ensemble.sel({self.responses[main_key].plot_settings.name:self.responses[main_key].plot_settings.key_type(sub_key)})["values"])
