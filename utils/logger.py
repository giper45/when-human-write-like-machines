
import logging
from pathlib import Path
import __main__


def _get_caller_script_name(default: str = "dataset-generation") -> str:
	main_file = getattr(__main__, "__file__", None)
	if not main_file:
		return default
	return Path(main_file).stem or default


NAME = _get_caller_script_name()
log = logging.getLogger(NAME)

