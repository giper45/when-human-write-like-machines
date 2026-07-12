

import hydra
from omegaconf import DictConfig
from utils.logger import log
from utils.evaluation.ResultsReportBuilder import (
    build_h2l_robustness_table,
    build_llm2l_robustness_table,
    build_source_origin_gap_table,
)

from utils.configuration import list_detector_model_names
from pathlib import Path

from utils.evaluation.Metrics import Metrics
from utils.evaluation.ResultsReportBuilder import build_detector_table


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    detectors = list_detector_model_names("conf", "detector")
    # print(detectors)
    results_dir = Path(cfg.experiment.paths.results_dir) 
    log.info(f"Build detector table for results_dir: {results_dir}")
    detector_table_df = build_detector_table(results_dir, 
                                        detectors, 
                                        baseline_regime="free_llm",
                                        n_bootstrap=5000, 
                                        random_seed=42)
    log.info(f"Detector table:\n{detector_table_df}")
    detector_table_df.to_csv(results_dir / "baseline_detector_table.csv")
    # 4.2 robustness under h2l rewriting
    log.info(f"Build H2L robustness table for results_dir: {results_dir}")
    detector_table = build_h2l_robustness_table(results_dir, detectors, n_bootstrap=5000, random_seed=42)
    detector_table.to_csv(results_dir / "h2l_robustness_table.csv")

    # 4.3 robustness under llm2l rewriting
    log.info(f"Build LLM2L robustness table for results_dir: {results_dir}")
    detector_table = build_llm2l_robustness_table(results_dir, detectors, n_bootstrap=5000, random_seed=42)
    detector_table.to_csv(results_dir / "llm2l_robustness_table.csv")

    log.info("Build paired LLM2L-minus-H2L source-origin gap table")
    detector_table = build_source_origin_gap_table(
        results_dir,
        detectors,
        n_bootstrap=5000,
        random_seed=42,
    )
    detector_table.to_csv(results_dir / "source_origin_gap_table.csv")

if __name__ == "__main__":
    main()
