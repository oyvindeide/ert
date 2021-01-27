import pathlib
import os
import shutil

from res.enkf import EnKFMain, ResConfig

from ert_data.measured import MeasuredData
from ert_shared.libres_facade import LibresFacade
from tests.utils import SOURCE_DIR

test_data_root = pathlib.Path(SOURCE_DIR) / "test-data" / "local"


def test_measured_data(monkeypatch, tmpdir):
    with tmpdir.as_cwd():
        test_data_dir = os.path.join(test_data_root, "snake_oil")

        shutil.copytree(test_data_dir, "test_data")
        os.chdir("test_data")

        obs_file = pathlib.Path.cwd() / "observations" / "observations.txt"
        with obs_file.open(mode='a') as fin:
            fin.write(create_summary_observation())

        res_config = ResConfig("snake_oil.ert")
        ert = EnKFMain(res_config)

        facade = LibresFacade(ert)
        import time

        start_time = time.time()
        MeasuredData(facade, [f"FOPR_{restart}" for restart in range(1, 200)])
        summary_obs_time = time.time() - start_time

        start_time = time.time()
        MeasuredData(facade, ["FOPR"])
        history_obs_time = time.time() - start_time


    print(1)


def create_summary_observation():
    import numpy as np
    observations = ""
    values = np.random.uniform(0,1.5,199)
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
