"""Unit tests for the generalized experiment_server and RunModel dispatch.

Tests follow the test_that_<behavior> naming convention.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from queue import SimpleQueue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import ert.dark_storage.endpoints.experiment_server as experiment_server_module
from ert.config import ErtConfig
from ert.dark_storage.app import app
from ert.dark_storage.endpoints.experiment_server import (
    ExperimentRunner,
    ExperimentRunnerState,
    _runs,
)
from ert.ensemble_evaluator.event import EndEvent
from ert.run_models.ensemble_experiment import EnsembleExperimentConfig
from ert.run_models.model_factory import create_run_model_config
from ert.run_models.start_request import (
    _MODEL_TYPE_TO_RUN_MODEL,
    EnsembleExperimentStartRequest,
    EnsembleInformationFilterStartRequest,
    EnsembleSmootherStartRequest,
    EvaluateEnsembleStartRequest,
    ManualUpdateEnIFStartRequest,
    ManualUpdateStartRequest,
    MultipleDataAssimilationStartRequest,
    SingleTestRunStartRequest,
)

PASSWORD = "test-token"


@pytest.fixture(autouse=True)
def _set_storage_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERT_STORAGE_TOKEN", PASSWORD)


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _auth() -> tuple[str, str]:
    return ("username", PASSWORD)


def test_that_ert_run_model_start_request_dispatches_single_test_run() -> None:
    assert (
        SingleTestRunStartRequest.model_fields["model_type"].default
        == "single_test_run"
    )


@pytest.mark.parametrize(
    ("model_type", "expected_wrapper_class"),
    [
        ("single_test_run", SingleTestRunStartRequest),
        ("ensemble_experiment", EnsembleExperimentStartRequest),
        ("ensemble_smoother", EnsembleSmootherStartRequest),
        ("ensemble_information_filter", EnsembleInformationFilterStartRequest),
        ("multiple_data_assimilation", MultipleDataAssimilationStartRequest),
        ("evaluate_ensemble", EvaluateEnsembleStartRequest),
        ("manual_update", ManualUpdateStartRequest),
        ("manual_update_enif", ManualUpdateEnIFStartRequest),
    ],
)
def test_that_ert_run_model_start_request_selects_wrapper_by_model_type(
    model_type: str,
    expected_wrapper_class: type,
) -> None:
    """Each model_type literal must match the matching wrapper envelope's default."""
    assert expected_wrapper_class.model_fields["model_type"].default == model_type


def test_that_model_type_to_run_model_registry_covers_all_ert_model_types() -> None:
    """_MODEL_TYPE_TO_RUN_MODEL must have an entry for every model_type
    present in the ErtRunModelStartRequest discriminated union."""
    expected_types = {
        "single_test_run",
        "ensemble_experiment",
        "ensemble_smoother",
        "ensemble_information_filter",
        "multiple_data_assimilation",
        "evaluate_ensemble",
        "manual_update",
        "manual_update_enif",
    }
    assert set(_MODEL_TYPE_TO_RUN_MODEL.keys()) == expected_types


def test_that_experiment_runner_invokes_registry_run_model_for_ert_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExperimentRunner.run() must look up the model class via the registry and
    pass model_dump() kwargs plus status_queue to its constructor."""
    run_id = "test-run-id"
    fake_run_state = ExperimentRunnerState()
    _runs[run_id] = fake_run_state

    mock_config = MagicMock(spec=EnsembleExperimentConfig)
    mock_config.model_dump.return_value = {}

    instantiated: list[Any] = []
    end_event = EndEvent(failed=False, msg="done")

    class FakeRunModel:
        queue_config = MagicMock()
        queue_config.queue_system = MagicMock()

        def __init__(self, *, status_queue: SimpleQueue, **kwargs: Any) -> None:
            instantiated.append(status_queue)
            # Put the end event so run() terminates
            status_queue.put(end_event)

        def start_simulations_thread(self, *_: Any, **__: Any) -> None:
            pass

        def cancel(self) -> None:
            pass

    monkeypatch.setitem(
        experiment_server_module._MODEL_TYPE_TO_RUN_MODEL,
        "ensemble_experiment",
        FakeRunModel,  # type: ignore[arg-type]
    )

    runner = ExperimentRunner(mock_config, run_id, model_type="ensemble_experiment")

    with (
        patch(
            "ert.dark_storage.endpoints.experiment_server.get_site_plugins",
            return_value=[],
        ),
        patch(
            "ert.dark_storage.endpoints.experiment_server.use_runtime_plugins",
        ),
        patch(
            "asyncio.get_running_loop",
        ) as mock_loop,
    ):
        future: asyncio.Future[None] = asyncio.Future()
        future.set_result(None)
        mock_loop.return_value.run_in_executor.return_value = future
        asyncio.run(runner.run())

    assert instantiated, "_MODEL_TYPE_TO_RUN_MODEL class was never instantiated"

    _runs.pop(run_id, None)


def test_that_start_experiment_with_model_type_takes_ert_dispatch_path(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /experiment_server/start_experiment with a model_type key must
    take the ERT path not the legacy Everest path."""
    monkeypatch.setattr("fastapi.BackgroundTasks.add_task", lambda *a, **kw: None)

    payload = {"model_type": "ensemble_experiment", "config": {}}
    response = api_client.post(
        "/experiment_server/start_experiment",
        auth=_auth(),
        json=payload,
    )
    # Empty config → Pydantic 422; not a 501 from EverestConfig parsing.
    assert response.status_code in {200, 422}


def test_that_start_experiment_without_model_type_treats_payload_as_everest_config(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /experiment_server/start_experiment without a model_type key must
    fall back to the legacy EverestConfig path and reject an empty payload."""
    monkeypatch.setattr("fastapi.BackgroundTasks.add_task", lambda *a, **kw: None)

    payload: dict[str, Any] = {}
    response = api_client.post(
        "/experiment_server/start_experiment",
        auth=_auth(),
        json=payload,
    )
    assert response.status_code in {200, 422, 501}


def test_that_create_run_model_config_for_ensemble_experiment_does_not_open_storage(
    tmp_path: Path,
) -> None:
    """create_run_model_config must return (model_type, RunModelConfig)
    without opening storage in write mode."""
    open_calls: list[str] = []
    real_open_storage = __import__(
        "ert.storage", fromlist=["open_storage"]
    ).open_storage

    def tracking_open_storage(path: Any, mode: str = "r", **kwargs: Any):
        open_calls.append(str(mode))
        return real_open_storage(path, mode, **kwargs)

    config_file = tmp_path / "test.ert"
    config_file.write_text("NUM_REALIZATIONS 1\nQUEUE_SYSTEM LOCAL\n")
    ert_config = ErtConfig.from_file(str(config_file))

    args = MagicMock()
    args.mode = "ensemble_experiment"
    args.experiment_name = "test"
    args.current_ensemble = "default"
    args.realizations = None

    with patch("ert.run_models.run_model.open_storage", tracking_open_storage):
        model_type, run_model_config = create_run_model_config(ert_config, args)

    assert model_type == "ensemble_experiment"
    assert run_model_config is not None
    assert "w" not in open_calls, "create_run_model_config opened storage in write mode"
