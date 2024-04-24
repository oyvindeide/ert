from unittest.mock import MagicMock

import pytest

from ert.config import Field, GenKwConfig, SurfaceConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.field_utils import FieldFileFormat
from ert.gui.ertwidgets.models.ertsummary import ErtSummary


@pytest.fixture
def mock_ert(monkeypatch):
    ert_mock = MagicMock()

    ert_mock.forward_model_step_name_list.return_value = [  # noqa
        "forward_model_1",
        "forward_model_2",
    ]

    gen_kw = GenKwConfig(
        name="KEY",
        forward_init=False,
        template_file="",
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
            TransformFunctionDefinition("KEY2", "NORMAL", [0, 1]),
            TransformFunctionDefinition("KEY3", "LOGNORMAL", [0, 1]),
        ],
        output_file="kw.txt",
        update=True,
    )

    surface = SurfaceConfig(
        "some_name",
        forward_init=True,
        ncol=10,
        nrow=7,
        xori=1,
        yori=1,
        xinc=1,
        yinc=1,
        rotation=1,
        yflip=1,
        forward_init_file="input_%d",
        output_file="output",
        base_surface_path="base_surface",
        update=True,
    )

    field = Field(
        name="some_name",
        forward_init=True,
        nx=10,
        ny=5,
        nz=3,
        file_format=FieldFileFormat.ROFF,
        output_transformation=None,
        input_transformation=None,
        truncation_min=None,
        truncation_max=None,
        forward_init_file="",
        output_file="",
        grid_file="",
        update=True,
    )

    ert_mock.ensemble_config.parameter_configs = {
        "surface": surface,
        "gen_kw": gen_kw,
        "field": field,
    }

    yield ert_mock


def test_getForwardModels(mock_ert):
    expected_list = ["forward_model_1", "forward_model_2"]
    forward_model_list = ErtSummary(mock_ert).getForwardModels()
    assert forward_model_list == expected_list


def test_getParameters(mock_ert):
    expected_list = ["field (10, 5, 3)", "gen_kw (3)", "surface (10, 7)"]
    parameter_list, parameter_count = ErtSummary(mock_ert).getParameters()
    assert parameter_list == expected_list
    assert parameter_count == 223


def test_snake_oil(snake_oil_case):
    summary = ErtSummary(snake_oil_case)

    assert summary.getForwardModels() == [
        "SNAKE_OIL_SIMULATOR",
        "SNAKE_OIL_NPV",
        "SNAKE_OIL_DIFF",
    ]

    assert summary.getParameters() == (["SNAKE_OIL_PARAM (10)"], 10)

    assert summary.getObservations() == [
        "FOPR",
        "WOPR_OP1_108",
        "WOPR_OP1_144",
        "WOPR_OP1_190",
        "WOPR_OP1_36",
        "WOPR_OP1_72",
        "WOPR_OP1_9",
        "WPR_DIFF_1",
    ]
