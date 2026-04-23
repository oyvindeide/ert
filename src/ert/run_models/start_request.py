"""Discriminated-union wrappers used by the experiment_server to deserialize
inbound ``start_experiment`` requests safely across all ERT RunModel types.

Each concrete ``*StartRequest`` class pairs a ``model_type`` discriminator
literal with the matching ``RunModelConfig`` subclass.  The server receives
one of these envelope objects, looks up the target ``RunModel`` class via
``_MODEL_TYPE_TO_RUN_MODEL``, and instantiates it from the nested config.

Everest experiments are handled separately (legacy path in experiment_server)
and are therefore **not** included here.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ert.run_models.ensemble_experiment import (
    EnsembleExperiment,
    EnsembleExperimentConfig,
)
from ert.run_models.ensemble_information_filter import (
    EnsembleInformationFilter,
    EnsembleInformationFilterConfig,
)
from ert.run_models.ensemble_smoother import EnsembleSmoother, EnsembleSmootherConfig
from ert.run_models.evaluate_ensemble import EvaluateEnsemble, EvaluateEnsembleConfig
from ert.run_models.manual_update import ManualUpdate, ManualUpdateConfig
from ert.run_models.manual_update_enif import ManualUpdateEnIF
from ert.run_models.multiple_data_assimilation import (
    MultipleDataAssimilation,
    MultipleDataAssimilationConfig,
)
from ert.run_models.run_model import RunModel
from ert.run_models.single_test_run import SingleTestRun, SingleTestRunConfig


class SingleTestRunStartRequest(BaseModel):
    model_type: Literal["single_test_run"] = "single_test_run"
    config: SingleTestRunConfig


class EnsembleExperimentStartRequest(BaseModel):
    model_type: Literal["ensemble_experiment"] = "ensemble_experiment"
    config: EnsembleExperimentConfig


class EnsembleSmootherStartRequest(BaseModel):
    model_type: Literal["ensemble_smoother"] = "ensemble_smoother"
    config: EnsembleSmootherConfig


class EnsembleInformationFilterStartRequest(BaseModel):
    model_type: Literal["ensemble_information_filter"] = "ensemble_information_filter"
    config: EnsembleInformationFilterConfig


class MultipleDataAssimilationStartRequest(BaseModel):
    model_type: Literal["multiple_data_assimilation"] = "multiple_data_assimilation"
    config: MultipleDataAssimilationConfig


class EvaluateEnsembleStartRequest(BaseModel):
    model_type: Literal["evaluate_ensemble"] = "evaluate_ensemble"
    config: EvaluateEnsembleConfig


class ManualUpdateStartRequest(BaseModel):
    model_type: Literal["manual_update"] = "manual_update"
    config: ManualUpdateConfig


class ManualUpdateEnIFStartRequest(BaseModel):
    """Manual EnIF update uses the same config shape as ManualUpdate."""

    model_type: Literal["manual_update_enif"] = "manual_update_enif"
    config: ManualUpdateConfig


ErtRunModelStartRequest = Annotated[
    SingleTestRunStartRequest
    | EnsembleExperimentStartRequest
    | EnsembleSmootherStartRequest
    | EnsembleInformationFilterStartRequest
    | MultipleDataAssimilationStartRequest
    | EvaluateEnsembleStartRequest
    | ManualUpdateStartRequest
    | ManualUpdateEnIFStartRequest,
    Field(discriminator="model_type"),
]

# Maps the model_type discriminator string to the concrete RunModel class.
# ManualUpdate and ManualUpdateEnIF both use ManualUpdateConfig but are
# distinguished by their model_type wrapper, so the key is the string.
_MODEL_TYPE_TO_RUN_MODEL: dict[str, type[RunModel]] = {
    "single_test_run": SingleTestRun,
    "ensemble_experiment": EnsembleExperiment,
    "ensemble_smoother": EnsembleSmoother,
    "ensemble_information_filter": EnsembleInformationFilter,
    "multiple_data_assimilation": MultipleDataAssimilation,
    "evaluate_ensemble": EvaluateEnsemble,
    "manual_update": ManualUpdate,
    "manual_update_enif": ManualUpdateEnIF,
}
