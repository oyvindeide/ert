from unittest.mock import MagicMock, call
from uuid import UUID

from ert._c_wrappers.enkf.enums import HookRuntime
from ert.shared.models import (
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
)

EXPECTED_CALL_ORDER = [
    HookRuntime.PRE_SIMULATION,
    HookRuntime.POST_SIMULATION,
    HookRuntime.PRE_FIRST_UPDATE,
    HookRuntime.PRE_UPDATE,
    HookRuntime.POST_UPDATE,
    HookRuntime.PRE_SIMULATION,
    HookRuntime.POST_SIMULATION,
]


def test_hook_call_order_ensemble_smoother(storage):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=0,
    )
    ert_mock.resConfig.return_value.preferred_job_fmt.return_value = "job_%d"
    minimum_args = {
        "current_case": "default",
        "active_realizations": [True],
        "target_case": "smooth",
    }
    test_class = EnsembleSmoother(
        minimum_args, ert_mock, storage, MagicMock(), UUID(int=0)
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


def test_hook_call_order_es_mda(storage):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    minimum_args = {
        "start_iteration": 0,
        "weights": "1",
        "num_iterations": 1,
        "analysis_module": "some_module",
        "active_realizations": [True],
        "current_case": "default",
        "target_case": "target_%d",
        "restart_run": False,
        "prior_ensemble": "",
    }
    ert_mock = MagicMock()
    ert_mock.ensemble_context.return_value.sim_fs.id = UUID(int=0)
    test_class = MultipleDataAssimilation(
        minimum_args, ert_mock, storage, MagicMock(), UUID(int=0), prior_ensemble=None
    )
    ert_mock.runWorkflows = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


class MockWContainer:
    def __init__(self):
        self.iteration_nr = 1


class MockEsUpdate:
    def iterative_smoother_update(self, _, posterior_storage, w_container, run_id):
        w_container.iteration_nr += 1


def test_hook_call_order_iterative_ensemble_smoother(storage):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=10,
    )
    ert_mock.ensemble_context.return_value.sim_fs.id = UUID(int=0)
    minimum_args = {
        "num_iterations": 1,
        "active_realizations": [True],
        "current_case": "default",
        "target_case": "target_%d",
    }

    test_class = IteratedEnsembleSmoother(
        minimum_args, ert_mock, storage, MagicMock(), UUID(int=0)
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.setPhase = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=1)
    test_class.facade._es_update = MockEsUpdate()
    test_class._w_container = MockWContainer()
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls
