import hydra
from omegaconf import DictConfig

from utils.evaluation.Metrics import Metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = Metrics.load_from_folder(cfg.experiment.paths.results_dir)
    Metrics.print_table(metrics)


if __name__ == "__main__":
    main()


