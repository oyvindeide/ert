import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import xarray as xr

from ert.config.parameter_config import CustomDict


@dataclasses.dataclass
class PlotSettings:
    name: str
    keys: List[any]
    key_type: str = str

@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data

    @property
    def plot_settings(self):
        return PlotSettings(name="name", keys=[self.name])
