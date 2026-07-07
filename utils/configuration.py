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