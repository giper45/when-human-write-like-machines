import atexit
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
import hydra
from utils.device import clear_gpu_full
from utils.dataset import DatasetLoader
from utils.configuration import is_pilot, setup_hf_logging
from utils.logger import log
from utils.text_filter import apply_text_filters

atexit.register(clear_gpu_full)


def pilot_execution(cfg):
	log.info("Pilot execution started...")
	# Add your pilot execution logic here
	dataset_loader = DatasetLoader(cfg.experiment.paths.dataset_home_path)
	dataset_loader.load(cfg.dataset.no_toxic_name)
	pilot_samples = cfg.experiment.pilot_length
	log.info(f"Selecting first {pilot_samples} samples from the dataset.")
	dataset = dataset_loader.get_first_n_samples(pilot_samples)
	dataset, filtering_report = apply_text_filters(
		dataset,
		cfg.experiment.filtering,
		output_dir=cfg.experiment.paths.output_dir,
		dataset_name=cfg.dataset.no_toxic_name,
	)
	log.info(f"Filtering complete: {filtering_report.rows_initial} → {filtering_report.rows_final} rows")
	dataset_loader.save(dataset, cfg.dataset.no_toxic_name_pilot)
	log.info("Pilot execution completed.")


def inspect(cfg):
	log.info("Inspection mode started...")
	# Add your inspection logic here
	dataset_loader = DatasetLoader(cfg.experiment.paths.dataset_home_path)
	dataset_loader.load(cfg.dataset.cleaned_dataset_name)
	print(dataset_loader.dataset)
	log.info("Inspection mode completed.")


def run_execution(cfg):
	log.info("Full execution started...")
	# Add your full execution logic here
	dataset_loader = DatasetLoader(cfg.experiment.paths.dataset_home_path)
	dataset = load_dataset(cfg.dataset.dataset_hf, split=cfg.dataset.split)
	if (cfg.dataset.text_column != "text"):
		log.info(f"Renaming column {cfg.dataset.text_column} to 'text'")
		dataset = dataset.rename_column(cfg.dataset.text_column, "text")
	dataset, filtering_report = apply_text_filters(
		dataset,
		cfg.experiment.filtering,
		output_dir=cfg.experiment.paths.output_dir,
		dataset_name=cfg.dataset.no_toxic_name,
	)
	log.info(f"Filtering complete: {filtering_report.rows_initial} → {filtering_report.rows_final} rows")
	dataset_loader.save(dataset, cfg.dataset.cleaned_dataset_name)
	log.info("Full execution completed.")	

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
	if is_pilot(cfg):
		pilot_execution(cfg)
	elif cfg.get("inspect", False):
		inspect(cfg)
	else:
		run_execution(cfg)

if __name__ == "__main__":
    main()

