"""In-process experiment client for GUI tests that don't have a server running.

Mirrors the interface of `ExperimentClient` but runs the experiment locally in
threads instead of POSTing to a server and subscribing via WebSocket.
"""

from __future__ import annotations

import threading
import uuid
from queue import SimpleQueue

from _ert.threading import ErtThread
from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import RunModelAPI, RunModelConfigUnion
from ert.run_models.event import StatusEvents
from ert.run_models.model_factory import _instantiate_run_model
from ert.run_models.run_model import RunModel

_SUPPORTS_RERUNNING: set[str] = {"EnsembleExperiment", "EvaluateEnsemble"}


class LocalExperimentClient:
    """Runs experiments in-process; used in tests instead of the HTTP client."""

    def __init__(self) -> None:
        self._event_queue: SimpleQueue[StatusEvents] | None = None
        self._run_model: RunModel | None = None
        self._run_id: str = str(uuid.uuid4())

    def start_experiment(
        self,
        config: RunModelConfigUnion,
        rerun_failed_realizations: bool = False,
    ) -> str:
        if self._event_queue is None:
            self._event_queue = SimpleQueue()
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()

        if rerun_failed_realizations and self._run_model is not None:
            run_model = self._run_model
            run_model._status_queue = status_queue
        else:
            run_model = _instantiate_run_model(config, status_queue)
            self._run_model = run_model
            self._run_id = str(uuid.uuid4())

        evaluator_server_config = (
            EvaluatorServerConfig()
            if run_model.queue_config.queue_system == QueueSystem.LOCAL
            else EvaluatorServerConfig(use_ipc_protocol=False)
        )
        event_queue = self._event_queue

        runner = threading.Thread(
            target=lambda: run_model.start_simulations_thread(
                evaluator_server_config,
                rerun_failed_realizations=rerun_failed_realizations,
            ),
            daemon=True,
            name="local_experiment_runner",
        )

        def _forward() -> None:
            while True:
                item = status_queue.get()
                event_queue.put(item)

        threading.Thread(
            target=_forward, daemon=True, name="local_event_bridge"
        ).start()
        runner.start()
        return self._run_id

    def setup_event_queue_from_ws_endpoint(
        self,
        run_id: str | None = None,
    ) -> tuple[SimpleQueue[StatusEvents], ErtThread]:
        if self._event_queue is None:
            self._event_queue = SimpleQueue()
        noop = ErtThread(target=lambda: None, daemon=True, name="local_noop_monitor")
        return self._event_queue, noop

    def create_run_model_api(
        self, config: RunModelConfigUnion | None = None
    ) -> RunModelAPI:
        if config is None:
            return RunModelAPI(
                experiment_name="",
                supports_rerunning_failed_realizations=False,
                cancel=self.stop,
                has_failed_realizations=lambda: False,
            )
        return RunModelAPI(
            experiment_name=config.model_type,
            supports_rerunning_failed_realizations=config.model_type
            in _SUPPORTS_RERUNNING,
            cancel=self.stop,
            has_failed_realizations=lambda: False,
        )

    def stop(self) -> None:
        if self._run_model is not None:
            self._run_model.cancel()
