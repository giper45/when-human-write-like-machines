import logging
from typing import List
from transformers.utils import logging as hf_logging

from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize
from omegaconf import OmegaConf



def setup_hf_logging():
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()
    hf_logging.disable_default_handler()



def is_pilot(cfg):
    return cfg.experiment.is_pilot

from pathlib import Path

def list_hydra_group_options(config_dir: str, group: str) -> list[str]:
    group_dir = Path(config_dir) / group

    if not group_dir.exists():
        raise FileNotFoundError(f"Config group not found: {group_dir}")

    return sorted(
        p.stem
        for p in group_dir.glob("*.yaml")
        if not p.name.startswith("_")
    )

from pathlib import Path
from omegaconf import OmegaConf


def list_detector_model_names(
    config_dir: str = "conf",
    group: str = "detectors",
) -> list[str]:
    group_dir = Path(config_dir) / group

    if not group_dir.exists():
        raise FileNotFoundError(f"Config group not found: {group_dir}")

    model_names = []

    for yaml_file in sorted(group_dir.glob("*.yaml")):
        if yaml_file.name.startswith("_"):
            continue

        cfg = OmegaConf.load(yaml_file)

        if "model_name" not in cfg:
            raise KeyError(f"Missing 'model_name' in {yaml_file}")

        model_names.append(cfg.model_name)

    return model_names

