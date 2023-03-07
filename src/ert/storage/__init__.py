from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import xtgeo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.enkf.config import FieldConfig


class Storage:
    def __init__(self, mount_point: Path, ensemble_size: int) -> None:
        self.mount_point = mount_point
        self.ensemble_size = ensemble_size

    def has_parameters(self) -> bool:
        """
        Checks if a parameter folder has been created
        """
        if Path(self.mount_point / "gen-kw.nc").exists():
            return True

        return False

    def save_gen_kw(  # pylint: disable=R0913
        self,
        parameter_name: str,
        parameter_keys: List[str],
        parameter_transfer_functions: List[Dict[str, Union[str, Dict[str, float]]]],
        realizations: List[int],
        data: npt.ArrayLike,
    ) -> None:
        if self.ensemble_size != len(realizations):
            padded_data = np.empty((len(parameter_keys), self.ensemble_size))
            for index, real in enumerate(realizations):
                padded_data[:, real] = data[:, index]
            data = padded_data

        ds = xr.Dataset(
            {
                parameter_name: ((f"{parameter_name}_keys", "iens"), data),
            },
            coords={
                f"{parameter_name}_keys": parameter_keys,
                "iens": range(self.ensemble_size),
            },
        )
        mode: Literal["a", "w"] = (
            "a" if Path.exists(self.mount_point / "gen-kw.nc") else "w"
        )

        ds.to_netcdf(self.mount_point / "gen-kw.nc", mode=mode, engine="scipy")
        priors = {}
        if Path.exists(self.mount_point / "gen-kw-priors.json"):
            with open(
                self.mount_point / "gen-kw-priors.json", "r", encoding="utf-8"
            ) as f:
                priors = json.load(f)
        priors.update({parameter_name: parameter_transfer_functions})
        with open(self.mount_point / "gen-kw-priors.json", "w", encoding="utf-8") as f:
            json.dump(priors, f)

    def load_gen_kw_realization(
        self, key: str, realization: int
    ) -> Tuple[npt.NDArray[np.double], List[str]]:
        input_file = self.mount_point / "gen-kw.nc"
        if not input_file.exists():
            raise KeyError(f"Unable to load GEN_KW for key: {key}")
        with xr.open_dataset(input_file, engine="scipy") as ds_disk:
            np_data = ds_disk.sel(iens=realization)[key].to_numpy()
            keys = list(ds_disk[key][f"{key}_keys"].values)

        return np_data, keys

    def save_summary_data(
        self,
        data: npt.NDArray[np.double],
        keys: List[str],
        axis: List[Any],
        realization: int,
    ) -> None:
        output_path = self.mount_point / f"summary-{realization}"
        Path.mkdir(output_path, exist_ok=True)

        ds = xr.Dataset(
            {"data": (("key", "time"), data)},
            coords={
                "key": keys,
                "time": axis,
            },
        )

        ds.to_netcdf(output_path / "data.nc", engine="scipy")

    def load_summary_data(
        self, summary_keys: List[str], realizations: List[int]
    ) -> Tuple[npt.NDArray[np.double], List[datetime], List[int]]:
        result = []
        loaded = []
        dates: List[datetime] = []
        for realization in realizations:
            input_path = self.mount_point / f"summary-{realization}"
            if not input_path.exists():
                continue
            loaded.append(realization)
            with xr.open_dataset(input_path / "data.nc", engine="scipy") as ds_disk:
                np_data = ds_disk["data"].to_numpy()
                keys = list(ds_disk["data"]["key"].values)
                if not dates:
                    dates = list(ds_disk["data"]["time"].values)
            indices = [keys.index(summary_key) for summary_key in summary_keys]
            selected_data = np_data[indices, :]

            result.append(selected_data.reshape(1, len(selected_data) * len(dates)).T)
        if not result:
            return np.array([]), dates, loaded
        return np.concatenate(result, axis=1), dates, loaded

    def load_summary_data_as_df(
        self, summary_keys: List[str], realizations: List[int]
    ) -> pd.DataFrame:
        data, time_axis, realizations = self.load_summary_data(
            summary_keys, realizations
        )
        if not data.any():
            raise KeyError(f"Unable to load SUMMARY_DATA for keys: {summary_keys}")
        multi_index = pd.MultiIndex.from_product(
            [summary_keys, time_axis], names=["data_key", "axis"]
        )
        return pd.DataFrame(
            data=data,
            index=multi_index,
            columns=realizations,
        )

    def save_gen_data(self, data: Dict[str, List[float]], realization: int) -> None:
        output_path = self.mount_point / f"gen-data-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        ds = xr.Dataset(
            data,
        )

        ds.to_netcdf(output_path / "data.nc", engine="scipy")

    def load_gen_data(
        self, key: str, realizations: List[int]
    ) -> Tuple[npt.NDArray[np.double], List[int]]:
        result = []
        loaded = []
        for realization in realizations:
            input_path = self.mount_point / f"gen-data-{realization}"
            if not input_path.exists():
                continue

            with xr.open_dataset(input_path / "data.nc", engine="scipy") as ds_disk:
                np_data = ds_disk[key].as_numpy()
                result.append(np_data)
                loaded.append(realization)
        if not result:
            raise KeyError(f"Unable to load GEN_DATA for key: {key}")
        return np.stack(result).T, loaded

    def load_gen_data_as_df(
        self, keys: List[str], realizations: List[int]
    ) -> pd.DataFrame:
        dfs = []
        for key in keys:
            data, realizations = self.load_gen_data(key, realizations)
            x_axis = [*range(data.shape[0])]
            multi_index = pd.MultiIndex.from_product(
                [[key], x_axis], names=["data_key", "axis"]
            )
            dfs.append(
                pd.DataFrame(
                    data=data,
                    index=multi_index,
                    columns=realizations,
                )
            )
        return pd.concat(dfs)

    def save_field_data(
        self,
        parameter_name: str,
        realization: int,
        data: npt.ArrayLike,
    ) -> None:
        output_path = self.mount_point / f"field-{realization}"
        Path.mkdir(output_path, exist_ok=True)
        np.save(f"{output_path}/{parameter_name}", data)

    def load_field(self, key: str, realizations: List[int]) -> npt.NDArray[np.double]:
        result = []
        for realization in realizations:
            input_path = self.mount_point / f"field-{realization}"
            if not input_path.exists():
                raise KeyError(f"Unable to load FIELD for key: {key}")
            data = np.load(input_path / f"{key}.npy")
            result.append(data)
        return np.stack(result).T  # type: ignore

    def field_has_data(self, key: str, realization: int) -> bool:
        path = self.mount_point / f"field-{realization}/{key}.npy"
        return path.exists()

    def export_field(
        self, config_node: FieldConfig, realization: int, output_path: str, fformat: str
    ) -> None:
        input_path = self.mount_point / f"field-{realization}"
        key = config_node.get_key()

        if not input_path.exists():
            raise KeyError(
                f"Unable to load FIELD for key: {key}, realization: {realization} "
            )
        data = np.load(input_path / f"{key}.npy")

        transform_name = config_node.get_output_transform_name()
        data_transformed = config_node.transform(transform_name, data)
        data_truncated = config_node.truncate(data_transformed)

        gp = xtgeo.GridProperty(
            ncol=config_node.get_nx(),
            nrow=config_node.get_ny(),
            nlay=config_node.get_nz(),
            values=data_truncated,
            grid=config_node.get_grid(),
            name=key,
        )

        os.makedirs(Path(output_path).parent, exist_ok=True)

        gp.to_file(output_path, fformat=fformat)

    def export_field_many(
        self,
        config_node: FieldConfig,
        realizations: List[int],
        output_path: str,
        fformat: str,
    ) -> None:
        for realization in realizations:
            file_name = output_path % realization
            try:
                self.export_field(config_node, realization, file_name, fformat)
                print(f"{config_node.get_key()}[{realization:03d}] -> {file_name}")
            except ValueError:
                sys.stderr.write(
                    f"ERROR: Could not load realisation:{realization} - export failed"
                )
                pass
