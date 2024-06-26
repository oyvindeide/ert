import json
from pathlib import Path
from uuid import UUID

from ert.storage.local_experiment import _Index

info = "Adding update property to parameters and removing template_file_path"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        key_to_remove = "template_file_path"
        for config in parameters_json.values():
            if key_to_remove in config:
                del config[key_to_remove]
            if "update" not in config:
                config["update"] = True
            if "name" not in config:
                config["name"] = "default_experiment_name"
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, indent=4))

        with open(experiment.joinpath("index.json"), mode="w", encoding="utf-8") as f:
            print(
                _Index(
                    id=UUID(experiment.name), name="default_experiment_name"
                ).model_dump_json(),
                file=f,
            )
        responses_file = experiment / "responses.json"
        with open(responses_file, encoding="utf-8", mode="r") as f:
            info = json.load(f)
        for key, values in list(info.items()):
            if values.get("_ert_kind") == "SummaryConfig" and not values.get("keys"):
                del info[key]
        with open(responses_file, encoding="utf-8", mode="w") as f:
            json.dump(info, f)
