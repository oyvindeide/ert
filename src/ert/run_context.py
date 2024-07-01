from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List

import numpy as np

from .run_arg import RunArg
from .runpaths import Runpaths

if TYPE_CHECKING:
    import numpy.typing as npt

    from .storage import Ensemble


def create_run_arguments(
    runpaths: Runpaths, active_realizations, iteration: int, ensemble: Ensemble
) -> List[RunArg]:
    run_args = []
    paths = runpaths.get_paths(list(range(len(active_realizations))), iteration)
    job_names = runpaths.get_jobnames(list(range(len(active_realizations))), iteration)

    for iens, (run_path, job_name, active) in enumerate(
        zip(paths, job_names, active_realizations)
    ):
        run_args.append(
            RunArg(
                str(ensemble.id),
                ensemble,
                iens,
                iteration,
                run_path,
                job_name,
                active,
            )
        )
    return run_args


@dataclass
class RunContext:
    ensemble: Ensemble
    runpaths: Runpaths
    initial_mask: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    iteration: int = 0

    def __post_init__(self) -> None:
        self.run_id = uuid.uuid4()
        self.run_args = create_run_arguments(
            self.runpaths, self.initial_mask, self.iteration, self.ensemble
        )
        self.runpaths.set_ert_ensemble(self.ensemble.name)

    @property
    def mask(self) -> List[bool]:
        return [real.active for real in self]

    def is_active(self, index: int) -> bool:
        return self[index].active

    @property
    def active_realizations(self) -> List[int]:
        return [i for i, real in enumerate(self) if real.active]

    def __len__(self) -> int:
        return len(self.initial_mask)

    def __getitem__(self, item: int) -> "RunArg":
        return self.run_args[item]

    def __iter__(self) -> Iterator["RunArg"]:
        yield from self.run_args

    def deactivate_realization(self, realization_nr: int) -> None:
        self[realization_nr].active = False
