from ert.gui.ertwidgets import resourceIcon
from ert.gui.tools import Tool
from ert.storage import open_storage

from .plot_window import PlotWindow


class PlotTool(Tool):
    def __init__(self, ens_path, main_window):
        super().__init__("Create plot", resourceIcon("timeline.svg"))
        self.main_window = main_window
        self.ens_path = ens_path

    def trigger(self):
        with open_storage(self.ens_path) as storage:
            plot_window = PlotWindow(storage, parent=self.main_window)
            plot_window.show()
