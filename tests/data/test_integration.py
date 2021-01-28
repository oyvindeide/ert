import time
import pathlib
import os
import shutil
import random

import numpy as np
from res.enkf import EnKFMain, ResConfig

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from tests.utils import SOURCE_DIR

test_data_root = pathlib.Path(SOURCE_DIR) / "test-data" / "local"


def test_summary_obs(monkeypatch, tmpdir):
    with tmpdir.as_cwd():
        test_data_dir = os.path.join(test_data_root, "snake_oil")

        shutil.copytree(test_data_dir, "test_data")
        os.chdir("test_data")

        obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
        with obs_file.open(mode="a") as fin:
            fin.write(create_summary_observation())

        res_config = ResConfig("snake_oil.ert")
        ert = EnKFMain(res_config)

        facade = LibresFacade(ert)

        start_time = time.time()
        foprh = MeasuredData(facade, [f"FOPR_{restart}" for restart in range(1, 200)])
        summary_obs_time = time.time() - start_time

        start_time = time.time()
        fopr = MeasuredData(facade, ["FOPR"])
        history_obs_time = time.time() - start_time

    assert summary_obs_time < 10 * history_obs_time

    result = foprh.get_simulated_data().values == fopr.get_simulated_data().values
    assert np.logical_and.reduce(result).all()


def test_data_gen_obs(monkeypatch, tmpdir):
    with tmpdir.as_cwd():
        test_data_dir = os.path.join(test_data_root, "snake_oil")

        shutil.copytree(test_data_dir, "test_data")
        os.chdir("test_data")

        obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
        with obs_file.open(mode="a") as fin:
            fin.write(create_general_observation())

        res_config = ResConfig("snake_oil.ert")
        ert = EnKFMain(res_config)

        facade = LibresFacade(ert)

        start_time = time.time()
        df = MeasuredData(facade, [f"CUSTOM_DIFF_{restart}" for restart in range(1, 500)])
        obs_time = time.time() - start_time

        df.remove_inactive_observations()
        assert df.data.shape == (27, 1995)


def create_summary_observation():
    observations = ""
    values = np.random.uniform(0, 1.5, 199)
    errors = values * 0.1
    for restart, (value, error) in enumerate(zip(values, errors)):
        restart += 1
        observations += f"""
    \nSUMMARY_OBSERVATION FOPR_{restart}
{{
    VALUE   = {value};
    ERROR   = {error};
    RESTART = {restart};
    KEY     = FOPR;
}};
    """
    return observations


def create_general_observation():
    observations = ""
    index_list = list(range(1, 2001))
    random.shuffle(index_list)
    index_list = [index_list[i:i + 4] for i in range(0, len(index_list), 4)]
    for nr, (i1, i2, i3, i4) in enumerate(index_list):
        observations += f"""
    \nGENERAL_OBSERVATION CUSTOM_DIFF_{nr}
{{
   DATA       = SNAKE_OIL_WPR_DIFF;
   INDEX_LIST = {i1},{i2},{i3},{i4};
   RESTART    = 199;
   OBS_FILE   = wpr_diff_obs.txt;
}};
    """
    return observations