from __future__ import annotations

import concurrent.futures
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QAction, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ert.dark_storage.client._session import find_conn_info
from ert.gui.detect_mode import is_dark_mode
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.find_ert_info import find_ert_info
from ert.gui.icon_utils import load_icon
from ert.gui.summarypanel import SummaryPanel
from ert.run_models import RunModel, create_run_model_config
from ert.run_models.run_model import RunModelConfig
from ert.runpaths import Runpaths

from .combobox_with_description import QComboBoxWithDescription
from .ensemble_experiment_panel import EnsembleExperimentPanel
from .ensemble_information_filter_panel import EnsembleInformationFilterPanel
from .ensemble_smoother_panel import EnsembleSmootherPanel
from .evaluate_ensemble_panel import EvaluateEnsemblePanel
from .experiment_client import ExperimentClient
from .experiment_config_panel import ExperimentConfigPanel
from .manual_update_panel import ManualUpdatePanel
from .multiple_data_assimilation_panel import MultipleDataAssimilationPanel
from .run_dialog import RunDialog
from .single_test_run_panel import SingleTestRunPanel
from .view.runpath_progress_widget import RunpathProgressWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig

EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE = "Execute selected"


def create_md_table(kv: dict[str, str], output: str) -> str:
    for key, unescaped_value in kv.items():
        value = unescaped_value.replace("_", r"\_")
        output += f"| {key} | {value} |\n"
    output += "\n"
    return output


def _runpath_check(
    run_model_config: RunModelConfig,
) -> tuple[bool, int, int]:
    """Return (runpath_exists, num_existing, num_active) using only the config."""
    run_paths = Runpaths(
        jobname_format=run_model_config.runpath_config.jobname_format_string,
        runpath_format=run_model_config.runpath_config.runpath_format_string,
        filename=str(run_model_config.runpath_file),
        substitutions=run_model_config.substitutions,
        eclbase=run_model_config.runpath_config.summary_file_base_name,
    )
    active = np.where(run_model_config.active_realizations)[0].tolist()
    paths = run_paths.get_paths(active, 0)  # iteration 0
    realization_dirs = {Path(p).parent for p in paths}
    num_existing = sum(1 for d in realization_dirs if d.exists())
    runpath_exists = any(Path(p).exists() for p in paths)
    num_active = run_model_config.active_realizations.count(True)
    return runpath_exists, num_existing, num_active


def _delete_runpaths(
    run_model_config: RunModelConfig,
    progress_tracker: RunpathProgressWidget | None = None,
    progress_callback: Any = None,
) -> None:
    """Delete run-path directories for all active realizations at iteration 0."""
    run_paths = Runpaths(
        jobname_format=run_model_config.runpath_config.jobname_format_string,
        runpath_format=run_model_config.runpath_config.runpath_format_string,
        filename=str(run_model_config.runpath_file),
        substitutions=run_model_config.substitutions,
        eclbase=run_model_config.runpath_config.summary_file_base_name,
    )
    active = np.where(run_model_config.active_realizations)[0].tolist()
    paths = run_paths.get_paths(active, 0)
    if progress_tracker is not None:
        progress_tracker.start(len(paths))
    if progress_callback is not None:
        progress_callback()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(lambda p: shutil.rmtree(p, ignore_errors=True), path)
            for path in paths
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            if progress_tracker is not None:
                progress_tracker.advance()
            if progress_callback is not None:
                progress_callback()


class ExperimentPanel(QWidget):
    experiment_type_changed = Signal(ExperimentConfigPanel)
    experiment_started = Signal(RunDialog)

    def __init__(
        self,
        config: ErtConfig,
        notifier: ErtNotifier,
        config_file: str,
    ) -> None:
        QWidget.__init__(self)
        self._notifier = notifier
        self.config = config
        run_path = config.runpath_config.runpath_format_string
        self._config_file = config_file

        self.setObjectName("experiment_panel")
        layout = QVBoxLayout()

        self._experiment_type_combo = QComboBoxWithDescription()
        self._experiment_type_combo.setObjectName("experiment_type")

        self._experiment_type_combo.currentIndexChanged.connect(
            self.toggleExperimentType
        )

        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.setContentsMargins(0, 0, 0, 0)
        experiment_type_layout.addWidget(
            self._experiment_type_combo, 0, Qt.AlignmentFlag.AlignVCenter
        )

        self._experiment_done: bool = True
        self.run_button = QToolButton()
        self.run_button.setObjectName("run_experiment")
        self.run_button.setIcon(load_icon("play_circle.svg"))
        self.run_button.setToolTip(EXPERIMENT_IS_MANUAL_UPDATE_MESSAGE)
        self.run_button.setIconSize(QSize(32, 32))
        self.run_button.clicked.connect(self.run_experiment)
        self.run_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.run_button.setMinimumWidth(60)
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet(
            """
            QToolButton {
            border-radius: 10px;
            background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #484848,
                stop:1 #323232
            );
            border: 1px solid #1e1e1e;
            padding: 5px;
            }
            QToolButton:hover {
                background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #575757,
                stop:1 #424242
            );
            }
        """
            if is_dark_mode()
            else """
            QToolButton {
            border-radius: 10px;
            background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #f0f0f0,
                stop:1 #d9d9d9
            );
            border: 1px solid #bfbfbf;
            padding: 5px;
            }
            QToolButton:hover {
                background-color: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #d8d8d8,
                stop:1 #c3c3c3
            );
            }
        """
        )

        experiment_type_layout.addWidget(self.run_button)
        experiment_type_layout.addStretch(1)

        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(experiment_type_layout)

        self._experiment_stack = QStackedWidget()
        self._experiment_stack.setLineWidth(1)
        self._experiment_stack.setFrameStyle(QFrame.Shape.StyledPanel)

        layout.addWidget(self._experiment_stack)

        self._experiment_widgets: dict[type[RunModel], ExperimentConfigPanel] = (
            OrderedDict()
        )
        analysis_config = config.analysis_config
        self.addExperimentConfigPanel(
            SingleTestRunPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
            ),
            True,
        )

        active_realizations = config.active_realizations
        config_num_realization = config.runpath_config.num_realizations
        self.addExperimentConfigPanel(
            EnsembleExperimentPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                active_realizations,
                config_num_realization,
                run_path,
                notifier,
            ),
            True,
        )
        self.addExperimentConfigPanel(
            EvaluateEnsemblePanel(run_path, notifier),
            True,
        )

        experiment_type_valid = any(
            p.update for p in config.ensemble_config.parameter_configs.values()
        ) and bool(config.observation_declarations)

        self.addExperimentConfigPanel(
            MultipleDataAssimilationPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            EnsembleSmootherPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            EnsembleInformationFilterPanel(
                analysis_config,
                config.ensemble_config.parameter_configuration,
                run_path,
                notifier,
                active_realizations,
                config_num_realization,
            ),
            experiment_type_valid,
        )
        self.addExperimentConfigPanel(
            ManualUpdatePanel(run_path, notifier, analysis_config),
            experiment_type_valid,
        )

        self.configuration_summary = SummaryPanel(config)
        layout.addWidget(self.configuration_summary)

        self.setLayout(layout)

    def addExperimentConfigPanel(
        self, panel: ExperimentConfigPanel, mode_enabled: bool
    ) -> None:
        assert isinstance(panel, ExperimentConfigPanel)
        self._experiment_stack.addWidget(panel)
        experiment_type = panel.get_experiment_type()
        self._experiment_widgets[experiment_type] = panel
        self._experiment_type_combo.addDescriptionItem(
            experiment_type.display_name(),
            experiment_type.description(),
            experiment_type.group(),
        )

        if not mode_enabled:
            item_count = self._experiment_type_combo.count() - 1
            model = self._experiment_type_combo.model()
            assert isinstance(model, QStandardItemModel)
            sim_item = model.item(item_count)
            assert sim_item is not None
            sim_item.setEnabled(False)
            sim_item.setToolTip(
                "Both observations and parameters must be defined.\n"
                "There must be parameters to update."
            )
            style = self.style()
            assert style is not None
            sim_item.setIcon(
                style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            )

        panel.experiment_configuration_changed.connect(self.validationStatusChanged)
        self.experiment_type_changed.connect(panel.experimentTypeChanged)

    @staticmethod
    def getActions() -> list[QAction]:
        return []

    def get_current_experiment_type(self) -> Any:
        experiment_type_display_name = self._experiment_type_combo.currentText()
        return next(
            w
            for w in self._experiment_widgets
            if w.display_name() == experiment_type_display_name
        )

    def get_experiment_arguments(self) -> Any:
        simulation_widget = self._experiment_widgets[self.get_current_experiment_type()]
        return simulation_widget.get_experiment_arguments()

    def run_experiment(self) -> None:
        args = self.get_experiment_arguments()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            _, run_model_config = create_run_model_config(
                self.config,
                args,
            )
        except ValueError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self,
                "ERROR: Failed to create experiment",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        QApplication.restoreOverrideCursor()

        runpath_exists, num_existing, num_active = _runpath_check(run_model_config)
        if runpath_exists:
            msg_box = QMessageBox(self)
            msg_box.setObjectName("RUN_PATH_WARNING_BOX")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setText("Run experiments")
            msg_box.setInformativeText(
                "ERT is running in an existing runpath.\n\n"
                "Please be aware of the following:\n"
                "- Previously generated results "
                "might be overwritten.\n"
                "- Previously generated files might "
                "be used if not configured correctly.\n"
                f"- {num_existing} out "
                f"of {num_active} realizations "
                "are running in existing runpaths.\n"
                "Are you sure you want to continue?"
            )
            delete_runpath_checkbox = QCheckBox()
            delete_runpath_checkbox.setText("Delete run_path")
            msg_box.setCheckBox(delete_runpath_checkbox)
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)
            msg_box.setWindowModality(Qt.WindowModality.ApplicationModal)
            msg_box_res = msg_box.exec()
            if msg_box_res == QMessageBox.StandardButton.No:
                return

            if delete_runpath_checkbox.checkState() == Qt.CheckState.Checked:
                progress_dialog = QDialog(self)
                progress_dialog.setObjectName("RUN_PATH_PROGRESS_DIALOG")
                progress_dialog.setWindowTitle("Deleting runpaths")
                progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
                progress_layout = QVBoxLayout(progress_dialog)
                progress_layout.setContentsMargins(0, 0, 0, 0)
                progress_widget = RunpathProgressWidget(
                    progress_dialog,
                    initial_status_text="Deleting runpaths...",
                    completed_action="deleted",
                )
                progress_layout.addWidget(progress_widget)
                progress_dialog.resize(420, 120)
                progress_dialog.show()
                QApplication.processEvents()
                try:
                    _delete_runpaths(
                        run_model_config,
                        progress_tracker=progress_widget,
                        progress_callback=QApplication.processEvents,
                    )
                except OSError as e:
                    progress_dialog.close()
                    progress_dialog.deleteLater()
                    err_box = QMessageBox(self)
                    err_box.setObjectName("RUN_PATH_ERROR_BOX")
                    err_box.setIcon(QMessageBox.Icon.Warning)
                    err_box.setText("ERT could not delete the existing runpath")
                    err_box.setInformativeText(
                        f"{e}\n\nContinue without deleting the runpath?"
                    )
                    err_box.setStandardButtons(
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    err_box.setDefaultButton(QMessageBox.StandardButton.No)
                    err_box.setWindowModality(Qt.WindowModality.ApplicationModal)
                    if err_box.exec() == QMessageBox.StandardButton.No:
                        return
                else:
                    progress_dialog.close()
                    progress_dialog.deleteLater()

        self.configuration_summary.log_summary(
            args.mode, run_model_config.active_realizations.count(True)
        )

        try:
            conn_info = find_conn_info()  # reads storage_server.json
            client = ExperimentClient.start_ert_experiment(conn_info, run_model_config)
        except Exception as e:
            QMessageBox.warning(
                self,
                "ERROR: Failed to start experiment",
                str(e),
                QMessageBox.StandardButton.Ok,
            )
            return

        event_queue, event_monitor_thread = client.setup_event_queue_from_ws_endpoint()
        run_model_api = client.create_run_model_api()

        self._dialog = RunDialog(
            f"Experiment - {self._config_file} {find_ert_info()}",
            run_model_api,
            event_queue,
            self._notifier,
            self.parent(),  # type: ignore
            output_path=self.config.analysis_config.log_path,
            run_path=Path(self.config.runpath_config.runpath_format_string),
            storage_path=self._notifier.storage.path,
        )
        self._dialog.queue_system.setText(
            "Queue system:\n"
            + run_model_config.queue_config.queue_system.formatted_name
        )
        self.experiment_started.emit(self._dialog)
        self._experiment_done = False
        self.run_button.setEnabled(self._experiment_done)

        event_monitor_thread.start()
        self._dialog.setup_event_monitoring()
        self._notifier.set_is_experiment_running(True)

        def simulation_done_handler() -> None:
            self._experiment_done = True
            self.run_button.setEnabled(self._experiment_done)
            self._notifier.emitErtChange()
            self.toggleExperimentType()

        self._dialog.experiment_done.connect(simulation_done_handler)

    def toggleExperimentType(self) -> None:
        current_model = self.get_current_experiment_type()
        if current_model is not None:
            widget = self._experiment_widgets[self.get_current_experiment_type()]
            self._experiment_stack.setCurrentWidget(widget)
            self.validationStatusChanged()
            self.experiment_type_changed.emit(widget)

    def validationStatusChanged(self) -> None:
        widget = self._experiment_widgets[self.get_current_experiment_type()]
        self.run_button.setEnabled(
            self._experiment_done and widget.isConfigurationValid()
        )
