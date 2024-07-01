from __future__ import annotations

from typing import TYPE_CHECKING, List

from .run_arg import RunArg
from .runpaths import Runpaths

if TYPE_CHECKING:
    from .storage import Ensemble


def create_run_arguments(
    runpaths: Runpaths,
    active_realizations: List[bool],
    iteration: int,
    ensemble: Ensemble,
) -> List[RunArg]:
    run_args = []
    runpaths.set_ert_ensemble(ensemble.name)
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
