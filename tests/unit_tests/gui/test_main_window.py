from typing import List

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from ert.gui.ertwidgets.analysismodulevariablespanel import AnalysisModuleVariablesPanel
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.tools.load_results.load_results_panel import LoadResultsPanel
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_case_selection_widget import CaseSelectionWidget
from ert.gui.tools.plot.plot_window import PlotWindow


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_plot_window_contains_the_expected_elements(
    esmda_has_run, opened_main_window, qtbot
):
    gui, _ = opened_main_window

    # Click on Create plot after esmda has run
    plot_tool = gui.tools["Create plot"]
    plot_tool.trigger()

    # Then the plot window opens
    qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
    plot_window = gui.findChild(PlotWindow)
    assert isinstance(plot_window, PlotWindow)

    case_selection = plot_window.findChild(CaseSelectionWidget)
    data_types = plot_window.findChild(DataTypeKeysWidget)
    assert isinstance(data_types, DataTypeKeysWidget)
    combo_boxes: List[QComboBox] = case_selection.findChildren(
        QComboBox
    )  # type: ignore

    # Assert that the Case selection widget contains the expected cases
    case_names = []
    assert len(combo_boxes) == 1
    combo_box = combo_boxes[0]
    for i in range(combo_box.count()):
        combo_box.setCurrentIndex(i)
        case_names.append(combo_box.currentText())
    assert sorted(case_names) == [
        "default",
        "default_0",
        "default_1",
        "default_2",
        "default_3",
    ]

    data_names = []
    data_keys = data_types.data_type_keys_widget
    for i in range(data_keys.model().rowCount()):
        index = data_keys.model().index(i, 0)
        data_names.append(str(index.data(Qt.DisplayRole)))
    assert data_names == [
        "POLY_RES@0",
        "COEFFS:COEFF_A",
        "COEFFS:COEFF_B",
        "COEFFS:COEFF_C",
    ]

    assert {
        plot_window._central_tab.tabText(i)
        for i in range(plot_window._central_tab.count())
    } == {
        "Cross case statistics",
        "Distribution",
        "Gaussian KDE",
        "Ensemble",
        "Histogram",
        "Statistics",
    }

    # Add all the cases
    for name in case_names:
        combo_box = combo_boxes[-1]
        for i in range(combo_box.count()):
            if combo_box.itemText(i) == name:
                combo_box.setCurrentIndex(i)
        qtbot.mouseClick(
            case_selection.findChild(QToolButton, name="add_case_button"), Qt.LeftButton
        )
        combo_boxes: List[QComboBox] = case_selection.findChildren(
            QComboBox
        )  # type: ignore
    assert len(case_selection.findChildren(QComboBox)) == len(case_names)

    # Cycle through showing all the tabs and plot each data key

    for i in range(data_keys.model().rowCount()):
        index = data_keys.model().index(i, 0)
        qtbot.mouseClick(
            data_types.data_type_keys_widget,
            Qt.LeftButton,
            pos=data_types.data_type_keys_widget.visualRect(index).center(),
        )
        for tab_index in range(plot_window._central_tab.count()):
            if not plot_window._central_tab.isTabEnabled(tab_index):
                continue
            plot_window._central_tab.setCurrentIndex(tab_index)
    plot_window.close()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_the_manage_cases_tool_can_be_used(
    esmda_has_run,
    opened_main_window,
    qtbot,
):
    gui, _ = opened_main_window

    # Click on "Manage Cases"
    def handle_dialog():
        qtbot.waitUntil(lambda: gui.findChild(ClosableDialog) is not None)
        dialog = gui.findChild(ClosableDialog)
        cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
        assert isinstance(cases_panel, CaseInitializationConfigurationPanel)

        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = current_tab.findChild(AddRemoveWidget)
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)

        # The case list should contain the expected cases
        assert case_list._list.count() == 5

        # Click add case and name it "new_case"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("new_case")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list should now contain "new_case"
        assert case_list._list.count() == 6

        # Go to the "initialize from scratch" panel
        cases_panel.setCurrentIndex(1)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "initialize_from_scratch_panel"
        combo_box = current_tab.findChild(CaseSelector)
        assert isinstance(combo_box, CaseSelector)

        # Select "new_case"
        current_index = 0
        while combo_box.currentText().startswith("new_case"):
            current_index += 1
            combo_box.setCurrentIndex(current_index)

        # click on "initialize"
        initialize_button = current_tab.findChild(
            QPushButton, name="initialize_from_scratch_button"
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()


@pytest.mark.usefixtures("use_tmpdir")
def test_that_inversion_type_can_be_set_from_gui(opened_main_window, qtbot):
    gui, _ = opened_main_window

    sim_mode = gui.findChild(QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)
    es_panel = gui.findChild(QWidget, name="ensemble_smoother_panel")
    es_edit = es_panel.findChild(QWidget, name="ensemble_smoother_edit")

    # Testing modal dialogs requires some care.
    # A helpful discussion on the topic is here:
    # https://github.com/pytest-dev/pytest-qt/issues/256
    def handle_dialog_first_time():
        var_panel = gui.findChild(AnalysisModuleVariablesPanel)
        inversion_spin_box = var_panel.findChild(QSpinBox, name="IES_INVERSION")
        assert inversion_spin_box.value() == 0
        qtbot.keyClick(inversion_spin_box, Qt.Key_Up)
        assert inversion_spin_box.value() == 1
        var_panel.parent().close()

    QTimer.singleShot(500, handle_dialog_first_time)
    qtbot.mouseClick(es_edit.findChild(QToolButton), Qt.LeftButton, delay=1)

    def handle_dialog_second_time():
        var_panel = gui.findChild(AnalysisModuleVariablesPanel)
        inversion_spin_box = var_panel.findChild(QSpinBox, name="IES_INVERSION")
        assert inversion_spin_box.value() == 1
        var_panel.parent().close()

    QTimer.singleShot(500, handle_dialog_second_time)
    qtbot.mouseClick(es_edit.findChild(QToolButton), Qt.LeftButton, delay=1)


def test_that_the_load_results_manually_tool_works(
    esmda_has_run, opened_main_window, qtbot
):
    gui, ensemble_size = opened_main_window

    def handle_load_results_dialog():
        qtbot.waitUntil(
           lambda: gui.findChild(ClosableDialog, name="load_results_manually_tool")
           is not None
        )
        dialog = gui.findChild(ClosableDialog, name="load_results_manually_tool")
        panel = dialog.findChild(LoadResultsPanel)
        assert isinstance(panel, LoadResultsPanel)

        case_selector = panel.findChild(CaseSelector)
        assert isinstance(case_selector, CaseSelector)
        index = case_selector.findText("default", Qt.MatchFlag.MatchContains)
        assert index != -1
        case_selector.setCurrentIndex(index)

        # click on "Load"
        load_button = panel.parent().findChild(QPushButton, name="Load")
        assert isinstance(load_button, QPushButton)

        # Verify that the messagebox is the success kind
        def handle_popup_dialog():
            messagebox = QApplication.activeModalWidget()
            assert isinstance(messagebox, QMessageBox)
            assert messagebox.text() == "Successfully loaded all realisations"
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_popup_dialog)
        qtbot.mouseClick(load_button, Qt.LeftButton)
        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    load_results_tool = gui.tools["Load results manually"]
    load_results_tool.trigger()


@pytest.mark.usefixtures("use_tmpdir")
def test_manage_cases_tool_with_clean_storage(opened_main_window_clean, qtbot):
    gui = opened_main_window_clean

    # Click on "Manage Cases"
    def handle_dialog():
        qtbot.waitUntil(lambda: gui.findChild(ClosableDialog) is not None)
        dialog = gui.findChild(ClosableDialog)
        cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
        assert isinstance(cases_panel, CaseInitializationConfigurationPanel)

        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = current_tab.findChild(AddRemoveWidget)
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)

        assert case_list._list.count() == 0

        # Click add case and name it "new_case"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("new_case")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        # The list should now contain "new_case"
        assert case_list._list.count() == 1
        assert case_list._list.item(0).data(Qt.UserRole).name == "new_case"

        # Go to the "initialize from scratch" panel
        cases_panel.setCurrentIndex(1)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "initialize_from_scratch_panel"
        combo_box = current_tab.findChild(CaseSelector)
        assert isinstance(combo_box, CaseSelector)

        assert combo_box.currentText().startswith("new_case")

        # click on "initialize"
        initialize_button = current_tab.findChild(
            QPushButton, name="initialize_from_scratch_button"
        )
        qtbot.mouseClick(initialize_button, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()
