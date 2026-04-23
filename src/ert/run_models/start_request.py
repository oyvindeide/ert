"""Discriminated union of all ERT RunModel config classes.

The ``model_type`` discriminator field is already defined on each concrete
``*Config`` class as a ``Literal[...]`` value.  ``ErtRunModelConfigUnion``
allows Pydantic (and FastAPI) to deserialize an inbound JSON object directly
into the correct config class without a separate envelope wrapper.

Everest experiments are handled separately (legacy path in experiment_server)
and are therefore **not** included here.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

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
from ert.run_models.manual_update_enif import ManualUpdateEnIF, ManualUpdateEnIFConfig
from ert.run_models.multiple_data_assimilation import (
    MultipleDataAssimilation,
    MultipleDataAssimilationConfig,
)
from ert.run_models.run_model import RunModel
from ert.run_models.single_test_run import SingleTestRun, SingleTestRunConfig

# Discriminated union of all concrete ERT run model config classes.
# FastAPI / Pydantic selects the correct member using the ``model_type``
# discriminator that each class declares as a ``Literal[...]`` field.
ErtRunModelConfigUnion = Annotated[
    SingleTestRunConfig
    | EnsembleExperimentConfig
    | EnsembleSmootherConfig
    | EnsembleInformationFilterConfig
    | MultipleDataAssimilationConfig
    | EvaluateEnsembleConfig
    | ManualUpdateConfig
    | ManualUpdateEnIFConfig,
    Field(discriminator="model_type"),
]

# Maps the model_type discriminator string to the concrete RunModel class.
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
