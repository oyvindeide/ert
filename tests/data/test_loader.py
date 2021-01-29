from ert_data import loader
from tests.data.mocked_block_observation import MockedBlockObservation
import sys
import pandas as pd
import pytest

from unittest.mock import Mock, MagicMock, ANY


def create_expected_data():
    return pd.DataFrame(
        [[10.0, 10.0, 10.0, 10.0]],
        index=[0],
    )


def create_summary_get_observations():
    return pd.DataFrame(
        [[10.0, None, 10.0, 10.0], [1.0, None, 1.0, 1.0]], index=["OBS", "STD"]
    )


def mocked_obs_node_get_index_nr(nr):
    return {0: 0, 1: 2, 2: 3}[nr]


@pytest.mark.parametrize(
    "obs_type,expected_loader",
    [
        ("GEN_OBS", (loader.load_general_data, loader.load_general_obs)),
        ("SUMMARY_OBS", (loader.load_summary_data, loader.load_summary_obs)),
        # ("BLOCK_OBS", loader.load_block_data),
    ],
)
def test_data_loader_factory(obs_type, expected_loader):
    assert loader.data_loader_factory(obs_type) == expected_loader


def test_data_loader_factory_fails():
    with pytest.raises(TypeError):
        loader.data_loader_factory("BAD_TYPE")


@pytest.mark.usefixtures("facade")
def test_load_general_data(facade, monkeypatch):
    mock_node = MagicMock()
    mock_node.__len__.return_value = 3
    mock_node.get_data_points.return_value = [10.0, 10.0, 10.0]
    mock_node.get_std.return_value = [1.0, 1.0, 1.0]
    mock_node.getIndex.side_effect = mocked_obs_node_get_index_nr

    facade.load_gen_data.return_value = pd.DataFrame(data=[10, 10, 10, 10])
    facade.get_observations()["some_key"].getNode.return_value = mock_node
    facade.all_data_type_keys.return_value = "some_key@1"
    facade.is_gen_data_key.return_value = True

    result = loader.load_general_data(facade, "some_key", "test_case")

    facade.load_gen_data.assert_called_once_with("test_case", "test_data_key", 1)
    mock_node.get_data_points.assert_called_once()
    mock_node.get_std.assert_called_once()

    assert result.equals(create_expected_data())


@pytest.mark.usefixtures("facade")
def test_load_block_data(facade, monkeypatch):
    mocked_get_block_measured = Mock(
        return_value=pd.DataFrame(data=[[10.0, 10.0, 10.0, 10.0]])
    )

    monkeypatch.setattr(loader, "_get_block_measured", mocked_get_block_measured)
    block_data = Mock()
    plot_block_data_loader = Mock()
    facade.create_plot_block_data_loader.return_value = plot_block_data_loader
    plot_block_data_loader.load.return_value = block_data

    mocked_block_obs = MockedBlockObservation(
        {"values": [10.0, None, 10.0, 10.0], "stds": [1.0, None, 1.0, 1.0]}
    )
    plot_block_data_loader.getBlockObservation.return_value = mocked_block_obs

    result = loader.load_block_data(facade, "some_key", "a_random_name")
    mocked_get_block_measured.assert_called_once_with(
        facade.get_ensemble_size(), block_data
    )
    assert result.equals(create_expected_data())


@pytest.mark.usefixtures("facade")
def test_load_summary_data(facade, monkeypatch):
    mocked_get_summary_observations = Mock(return_value=pd.DataFrame())
    mocked_get_summary_data = Mock(return_value=create_expected_data())


    monkeypatch.setattr(
        loader, "_get_summary_observations", mocked_get_summary_observations
    )
    monkeypatch.setattr(loader, "_get_summary_data", mocked_get_summary_data)

    data_key = "some_key"
    case_name = "a_random_name"

    result = loader.load_summary_data(facade, data_key, case_name)

    mocked_get_summary_data.assert_called_once_with(
        facade, data_key, case_name
    )
    assert result.equals(create_expected_data())


@pytest.mark.usefixtures("facade")
def test_load_summary_obs(facade, monkeypatch):
    mocked_get_summary_obs = Mock(return_value=create_summary_get_observations())
    mocked_remove_inactive_report_steps = Mock(
        return_value=create_summary_get_observations().dropna(axis=1)
    )

    monkeypatch.setattr(
        loader, "_get_summary_observations", mocked_get_summary_obs
    )
    monkeypatch.setattr(
        loader, "_remove_inactive_report_steps", mocked_remove_inactive_report_steps
    )

    data_key = "some_key"
    case_name = "a_random_name"

    result = loader.load_summary_obs(facade, [data_key], case_name)

    mocked_get_summary_obs.assert_called_once_with(
        facade, data_key, case_name
    )

    assert result["some_key"].equals(create_summary_get_observations().dropna(axis=1))
