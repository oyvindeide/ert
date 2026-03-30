import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import queue
import shutil
import time
import traceback
import uuid
from base64 import b64decode
from pathlib import Path
from queue import SimpleQueue
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocketException,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette import status
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from ert.config import QueueSystem
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.ensemble_evaluator.event import FullSnapshotEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.run_models import (
    RunModel,
    RunModelConfigUnion,
    StatusEvents,
    _compute_run_paths,
)
from ert.run_models.everest_run_model import (
    EverestExitCode,
    EverestRunModel,
    EverestRunModelConfig,
)
from ert.run_models.model_factory import _instantiate_run_model
from everest.detached.everserver import (
    ExperimentState,
    ExperimentStatus,
)
from everest.strings import (
    EXPERIMENT_SERVER,
    OPT_FAILURE_ALL_REALIZATIONS,
    OPT_FAILURE_REALIZATIONS,
    EverEndpoints,
)

router = APIRouter(prefix="/experiment_server", tags=["experiment_server"])


def _delete_runpath(run_path: str) -> None:
    if Path(run_path).exists():
        shutil.rmtree(run_path)


class UserCancelled(Exception):
    pass


@dataclasses.dataclass
class ExperimentRunnerState:
    status: ExperimentStatus = dataclasses.field(default_factory=ExperimentStatus)
    events: list[StatusEvents] = dataclasses.field(default_factory=list)
    subscribers: dict[str, "Subscriber"] = dataclasses.field(default_factory=dict)
    config_path: str | os.PathLike[str] | None = None
    run_path: str | os.PathLike[str] | None = None
    storage_path: str | os.PathLike[str] | None = None
    start_time_unix: int | None = None
    run_model: RunModel | EverestRunModel | None = None

    def reset(self) -> None:
        self.status = ExperimentStatus()
        self.events = []
        self.subscribers = {}
        self.config_path = None
        self.run_path = None
        self.storage_path = None
        self.start_time_unix = None
        self.run_model = None


_runs: dict[str, ExperimentRunnerState] = {}
_state: dict[str, str | None] = {"current_run_id": None}
security = HTTPBasic()


def _current_run() -> ExperimentRunnerState:
    run_id = _state["current_run_id"]
    if run_id is None or run_id not in _runs:
        return ExperimentRunnerState()
    return _runs[run_id]


def _failed_realizations_messages(
    events: list[StatusEvents], exit_code: EverestExitCode
) -> list[str]:
    snapshots: dict[int, EnsembleSnapshot] = {}
    for event in events:
        if isinstance(event, FullSnapshotEvent) and event.snapshot:
            snapshots[event.iteration] = event.snapshot
        elif isinstance(event, SnapshotUpdateEvent) and event.snapshot:
            snapshot = snapshots[event.iteration]
            assert isinstance(snapshot, EnsembleSnapshot)
            snapshot.merge_snapshot(event.snapshot)
    logging.getLogger("forward_models").info("Status event")
    messages = [
        OPT_FAILURE_REALIZATIONS
        if exit_code == EverestExitCode.TOO_FEW_REALIZATIONS
        else OPT_FAILURE_ALL_REALIZATIONS
    ]
    for snapshot in snapshots.values():
        for job in snapshot.get_all_fm_steps().values():
            if error := job.get("error"):
                msg = f"{job.get('name', 'Unknown name')} Failed with: {error}"
                if msg not in messages:
                    messages.append(msg)
    return messages


def _get_optimization_status(
    exit_code: EverestExitCode, events: list[StatusEvents]
) -> tuple[ExperimentState, str]:
    match exit_code:
        case EverestExitCode.MAX_BATCH_NUM_REACHED:
            return ExperimentState.completed, "Maximum number of batches reached."

        case EverestExitCode.MAX_FUNCTIONS_REACHED:
            return (
                ExperimentState.completed,
                "Maximum number of function evaluations reached.",
            )

        case EverestExitCode.USER_ABORT:
            return ExperimentState.stopped, "Optimization aborted."

        case (
            EverestExitCode.TOO_FEW_REALIZATIONS
            | EverestExitCode.ALL_REALIZATIONS_FAILED
        ):
            status_ = ExperimentState.failed
            messages = _failed_realizations_messages(events, exit_code)
            for msg in messages:
                logging.getLogger(EXPERIMENT_SERVER).error(msg)
            return status_, "\n".join(messages)
        case _:
            return ExperimentState.completed, "Optimization completed."


def _check_authentication(auth_header: str | None) -> None:
    if auth_header is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="No authentication"
        )
    _, encoded_credentials = auth_header.split(" ")
    decoded_credentials = b64decode(encoded_credentials).decode("utf-8")
    _, _, password = decoded_credentials.partition(":")
    if password != os.environ["ERT_STORAGE_TOKEN"]:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


def _check_user(credentials: HTTPBasicCredentials) -> None:
    if credentials.password != os.environ["ERT_STORAGE_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def _log(request: Request) -> None:
    logging.getLogger(EXPERIMENT_SERVER).debug(
        f"{request.scope['path']} entered from "
        f"{request.client.host if request.client else 'unknown host'} "
        f"with HTTP {request.method}"
    )


@router.get("/")
def get_status(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> PlainTextResponse:
    _log(request)
    _check_user(credentials)
    return PlainTextResponse("EVEREST is running")


@router.get("/status")
def experiment_status(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> ExperimentStatus:
    _log(request)
    _check_user(credentials)
    return _current_run().status


@router.post("/" + EverEndpoints.stop)
def stop(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> Response:
    _log(request)
    _check_user(credentials)
    run = _current_run()
    run.status = ExperimentStatus(
        message="Server stopped by user", status=ExperimentState.stopped
    )
    if run.run_model is not None:
        run.run_model.cancel()
    return Response("Raise STOP flag succeeded. EVEREST initiates shutdown..", 200)


@router.post("/" + EverEndpoints.start_experiment)
async def start_experiment(
    request: Request,
    background_tasks: BackgroundTasks,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    config: RunModelConfigUnion,
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    run_id = str(uuid.uuid4())
    run_state = ExperimentRunnerState()
    _runs[run_id] = run_state
    _state["current_run_id"] = run_id
    runner = ExperimentRunner(config, run_id)
    try:
        background_tasks.add_task(runner.run)
        if isinstance(config, EverestRunModelConfig):
            ec = config.everest_config
            run_state.config_path = ec.config_file
            run_state.run_path = ec.output_dir
            run_state.storage_path = str(ec.storage_dir)
        else:
            run_state.config_path = config.user_config_file
            run_state.run_path = config.runpath_config.runpath_format_string
            run_state.storage_path = config.storage_path

        run_state.start_time_unix = int(time.time())
        return JSONResponse({"run_id": run_id})
    except Exception as e:
        run_state.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=f"Could not start experiment: {e!s}",
        )
        logging.getLogger(EXPERIMENT_SERVER).exception(e)
        return JSONResponse(
            {"error": f"Could not start experiment: {e!s}"}, status_code=501
        )


@router.get("/" + EverEndpoints.config_path)
async def config_path(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    run = _current_run()
    if run.status.status == ExperimentState.pending:
        return JSONResponse("No experiment started", status_code=404)

    return JSONResponse(
        {
            "config_path": str(run.config_path),
            "run_path": str(run.run_path),
            "storage_path": str(run.storage_path),
        },
        status_code=200,
    )


@router.get("/" + EverEndpoints.start_time)
async def start_time(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> Response:
    _log(request)
    _check_user(credentials)
    run = _current_run()
    if run.status.status == ExperimentState.pending:
        return Response("No experiment started", status_code=404)

    return Response(str(run.start_time_unix), status_code=200)


@router.post("/check_runpath")
def check_runpath(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    config: RunModelConfigUnion,
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    paths = _compute_run_paths(config)
    realization_dirs = {Path(p).parent for p in paths}
    existing_count = sum(1 for d in realization_dirs if d.exists())
    active_count = sum(config.active_realizations)
    return JSONResponse(
        {"existing_count": existing_count, "active_count": active_count}
    )


@router.post("/delete_runpath")
def delete_runpath_endpoint(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    config: RunModelConfigUnion,
) -> Response:
    _log(request)
    _check_user(credentials)
    paths = _compute_run_paths(config)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(_delete_runpath, paths)
    return Response("Runpath deleted", status_code=200)


@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket, run_id: str | None = None) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    if run_id is None:
        try:
            run_id = next(reversed(_runs))
        except StopIteration:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    if run_id not in _runs:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    subscriber_id = str(uuid.uuid4())
    try:
        while True:
            event = await _get_event(subscriber_id=subscriber_id, run_id=run_id)
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
                break
    except Exception as e:
        logging.getLogger(EXPERIMENT_SERVER).exception(str(e))
    finally:
        logging.getLogger(EXPERIMENT_SERVER).info(
            f"Subscriber {subscriber_id} done. Closing websocket"
        )
        # Give some time for subscribers to get events
        await asyncio.sleep(5)
        _runs[run_id].subscribers[subscriber_id].done()


async def _get_event(subscriber_id: str, run_id: str) -> StatusEvents:
    """
    The function waits until there is an event available for the subscriber
    and returns the event. If the subscriber is up to date it will
    wait until we wake up the subscriber using notify
    """
    run = _runs[run_id]
    if subscriber_id not in run.subscribers:
        run.subscribers[subscriber_id] = Subscriber()
    subscriber = run.subscribers[subscriber_id]

    while subscriber.index >= len(run.events):
        await subscriber.wait_for_event()

    event = run.events[subscriber.index]
    subscriber.index += 1
    return event


def _get_final_status(
    run_model: RunModel, events: list[StatusEvents]
) -> ExperimentStatus:
    if isinstance(run_model, EverestRunModel):
        assert run_model.exit_code is not None
        exp_state, msg = _get_optimization_status(run_model.exit_code, events)
        return ExperimentStatus(message=msg, status=exp_state)
    return ExperimentStatus(
        message="Experiment completed.", status=ExperimentState.completed
    )


class ExperimentRunner:
    def __init__(
        self,
        config: RunModelConfigUnion,
        run_id: str,
    ) -> None:
        super().__init__()
        self._config = config
        self._run_id = run_id

    async def run(self) -> None:
        run = _runs[self._run_id]
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        run_model: RunModel | None = None
        try:
            run_model = _instantiate_run_model(self._config, status_queue)
            evaluator_server_config = (
                EvaluatorServerConfig()
                if run_model.queue_config.queue_system == QueueSystem.LOCAL
                else EvaluatorServerConfig(use_ipc_protocol=False)
            )
            run.run_model = run_model
            run.status = ExperimentStatus(
                message="Experiment started", status=ExperimentState.running
            )
            loop = asyncio.get_running_loop()
            simulation_future = loop.run_in_executor(
                None,
                lambda: run_model.start_simulations_thread(evaluator_server_config),
            )
            while True:
                if run.status.status == ExperimentState.stopped:
                    run_model.cancel()
                    raise UserCancelled("Experiment aborted")
                try:
                    item: StatusEvents = status_queue.get(block=False)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                run.events.append(item)
                for sub in run.subscribers.values():
                    sub.notify()

                if isinstance(item, EndEvent):
                    # Wait for subscribers to receive final events
                    for sub in list(run.subscribers.values()):
                        await sub.is_done()
                    break
            await simulation_future

            run.status = _get_final_status(run_model, run.events)
        except UserCancelled as e:
            logging.getLogger(EXPERIMENT_SERVER).info(f"User cancelled: {e}")
        except Exception as e:
            logging.getLogger(EXPERIMENT_SERVER).exception(e)
            run.status = ExperimentStatus(
                message=f"Exception: {e}\n{traceback.format_exc()}",
                status=ExperimentState.failed,
            )
        finally:
            if (
                isinstance(run_model, EverestRunModel)
                and run_model._experiment is not None
            ):
                run_model._experiment.status = run.status

            logging.getLogger(EXPERIMENT_SERVER).info(
                f"ExperimentRunner done. Items left in queue: {status_queue.qsize()}"
            )


class Subscriber:
    """
    This class keeps track of events and allows subscribers
    to wait for new events to occur. Each subscriber instance
    can be notified of an event, at which point any coroutines
    that are waiting for an event will resume execution.
    """

    def __init__(self) -> None:
        self.index = 0
        self._event = asyncio.Event()
        self._done = asyncio.Event()

    def notify(self) -> None:
        self._event.set()

    def done(self) -> None:
        self._done.set()

    async def wait_for_event(self) -> None:
        await self._event.wait()
        self._event.clear()

    async def is_done(self) -> None:
        await self._done.wait()
