from __future__ import annotations

import logging
import queue
import ssl
import time
import traceback
from base64 import b64encode
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from pydantic import TypeAdapter, ValidationError
from requests import HTTPError
from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from _ert.threading import ErtThread
from ert.dark_storage.client import ErtClientConnectionInfo
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import RunModelAPI
from ert.run_models.event import StatusEvents, status_event_from_json
from ert.run_models.start_request import ErtRunModelStartRequest
from everest.strings import EverEndpoints

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _build_ssl_context(cert_file: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=cert_file)
    return ctx


class ExperimentClient:
    def __init__(
        self,
        run_id: str,
        url: str,
        cert_file: str,
        username: str,
        password: str,
        ssl_context: ssl.SSLContext,
    ) -> None:
        self._run_id = run_id
        self._url = url
        self._cert = cert_file
        self._username = username
        self._password = password
        self._ssl_context = ssl_context

        self._is_alive = False
        self._start_time: int | None = None

    @classmethod
    def start_ert_experiment(
        cls,
        conn_info: ErtClientConnectionInfo,
        start_request: ErtRunModelStartRequest,
    ) -> ExperimentClient:
        """POST an ERT run-model config to the experiment_server and return a
        client already bound to the resulting run_id.

        Args:
            conn_info: Connection info for the running ERT storage server.
            start_request: A validated ``ErtRunModelStartRequest`` wrapping the
                serializable ``RunModelConfig``.

        Returns:
            An ``ExperimentClient`` ready to subscribe to events and cancel the run.
        """
        if not isinstance(conn_info.cert, str):
            raise RuntimeError("cert in conn_info must be a file path string")
        auth_token = conn_info.auth_token
        if auth_token is None:
            raise RuntimeError("No auth token found in storage connection info")

        url = conn_info.base_url.rstrip("/") + "/experiment_server"
        cert_file: str = conn_info.cert
        username = "username"
        password = auth_token

        adapter: TypeAdapter[ErtRunModelStartRequest] = TypeAdapter(
            ErtRunModelStartRequest
        )
        payload = adapter.dump_python(start_request, mode="json")

        response = requests.post(
            f"{url}/{EverEndpoints.start_experiment}",
            verify=cert_file,
            auth=(username, password),
            proxies={"http": None, "https": None},  # type: ignore[dict-item]
            json=payload,
        )
        response.raise_for_status()
        run_id: str = response.json()["run_id"]

        return cls(
            run_id=run_id,
            url=url,
            cert_file=cert_file,
            username=username,
            password=password,
            ssl_context=_build_ssl_context(cert_file),
        )

    def _http_get(self, endpoint: str) -> requests.Response:
        return requests.get(
            f"{self._url}/{endpoint}",
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore[dict-item]
        )

    def _http_post(self, endpoint: str) -> requests.Response:
        return requests.post(
            f"{self._url}/{endpoint}",
            verify=self._cert,
            auth=(self._username, self._password),
            proxies={"http": None, "https": None},  # type: ignore[dict-item]
        )

    @property
    def config(self) -> dict[str, str]:
        return self._http_get(f"{EverEndpoints.config_path}/{self._run_id}").json()

    @property
    def credentials(self) -> str:
        return b64encode(f"{self._username}:{self._password}".encode()).decode()

    def setup_event_queue_from_ws_endpoint(
        self,
        refresh_interval: float = 0.01,
        open_timeout: float = 30,
        websocket_recv_timeout: float = 1.0,
    ) -> tuple[queue.SimpleQueue[StatusEvents], ErtThread]:
        event_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

        def passthrough_ws_events() -> None:
            try:
                with connect(
                    self._url.replace("https://", "wss://")
                    + f"/{EverEndpoints.events}/{self._run_id}",
                    ssl=self._ssl_context,
                    open_timeout=open_timeout,
                    additional_headers={"Authorization": f"Basic {self.credentials}"},
                ) as websocket:
                    while not self._is_alive:
                        try:
                            message = websocket.recv(timeout=websocket_recv_timeout)
                        except TimeoutError:
                            message = None
                        if message:
                            try:
                                event = status_event_from_json(message)
                                event_queue.put(event)
                            except ValidationError as e:
                                logger.error(
                                    "Error when processing event %s", exc_info=e
                                )

                        time.sleep(refresh_interval)
            except ConnectionClosedError:
                logger.debug("Connection closed by server")
            except Exception:
                logger.debug(traceback.format_exc())

        monitor_thread = ErtThread(
            name="everest_gui_event_monitor",
            target=passthrough_ws_events,
            daemon=True,
        )

        return event_queue, monitor_thread

    def create_run_model_api(self) -> RunModelAPI:
        def start_fn(
            evaluator_server_config: EvaluatorServerConfig,
            rerun_failed_realizations: bool = False,
        ) -> None:
            pass

        return RunModelAPI(
            experiment_name=Path(self.config["config_path"]).name,
            supports_rerunning_failed_realizations=False,
            start_simulations_thread=start_fn,
            cancel=self.stop,
            has_failed_realizations=lambda: False,
        )

    def stop(self) -> None:
        try:
            response = self._http_post(EverEndpoints.stop)

            if response.status_code == 200:
                logger.info("Cancelled experiment from EVEREST")
                print("Successfully cancelled experiment")
            else:
                logger.error(
                    f"Failed to cancel EVEREST experiment: "
                    f"POST @ {self._url}/{EverEndpoints.stop}, "
                    f"server responded with status {response.status_code}: "
                    f"{HTTPStatus(response.status_code).phrase}"
                )
                print("Failed to cancel experiment")

        except requests.exceptions.ConnectionError as e:
            logger.error(
                "Connection error when cancelling EVEREST "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")

        except HTTPError as e:
            logger.error(
                "HTTP error when cancelling EVEREST "
                f"experiment: {''.join(traceback.format_exception(e))}"
            )
            print("Failed to cancel experiment")
