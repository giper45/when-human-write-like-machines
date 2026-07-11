from pathlib import Path
import sys

import hydra
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig

from utils.detector import load_detector_model
from utils.evaluation.Metrics import Metrics
from utils.logger import log


REGIME_ALIASES = {
    "freellm": "free_llm",
    "llmfree": "free_llm",
}

REGIME_FILE_ALIASES = {
    "free_llm": ["free_llm", "freellm", "llmfree"],
}


def normalize_regime_name(regime: str) -> str:
    if regime is None:
        return None
    return REGIME_ALIASES.get(str(regime), str(regime))


def load_machine_entries(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def get_detector_name(cfg: DictConfig) -> str:
    if "model_name" in cfg.detector:
        return cfg.detector.model_name
    if "observer_name_or_path" in cfg.detector:
        return "binoculars"
    return cfg.detector.detector_type


def get_target_regime(cfg: DictConfig) -> str:
    return normalize_regime_name(cfg.get("machine_postfix"))


def get_machine_texts_path(cfg: DictConfig) -> Path:
    configured_path = Path(cfg.experiment.paths.machine_file)
    if configured_path.exists():
        return configured_path

    target_regime = get_target_regime(cfg)
    generated_dir = Path(cfg.experiment.paths.generated_dir)
    for file_regime in REGIME_FILE_ALIASES.get(target_regime, [target_regime]):
        candidate = generated_dir / f"{cfg.dataset.name}_{cfg.model.name}_{file_regime}.txt"
        if candidate.exists():
            return candidate

    return configured_path


def get_tpr_fpr_target(cfg: DictConfig) -> float:
    evaluation_cfg = getattr(cfg.experiment, "evaluation", None)
    fpr_targets = getattr(evaluation_cfg, "fpr_targets", None) if evaluation_cfg else None
    if fpr_targets:
        return float(fpr_targets[0])
    return 0.01


def validate(cfg: DictConfig) -> None:
    allowed = {"h2l", "free_llm", "freellm", "llmfree", "llm2l"}
    if cfg.get("machine_postfix") not in allowed:
        raise ValueError(
            f"Invalid machine_postfix: {cfg.get('machine_postfix')}. "
            f"Allowed values are: {sorted(allowed)}"
        )


def build_detector_dataset(cfg: DictConfig):
    sampled_dataset_path = Path(cfg.experiment.paths.sampled_file)
    machine_texts_path = get_machine_texts_path(cfg)

    if not sampled_dataset_path.exists():
        raise FileNotFoundError(f"Sampled dataset not found: {sampled_dataset_path}")
    if not machine_texts_path.exists():
        raise FileNotFoundError(f"Machine texts file not found: {machine_texts_path}")

    sampled = load_from_disk(str(sampled_dataset_path))
    human_entries = list(sampled["text"])
    machine_entries = load_machine_entries(machine_texts_path)
    if len(machine_entries) != len(human_entries):
        raise ValueError(
            "Human/machine row count mismatch: "
            f"human={len(human_entries)}, machine={len(machine_entries)}, "
            f"machine_file={machine_texts_path}. Positional IDs would be invalid."
        )

    human_ids = [f"human::{idx}" for idx in range(len(human_entries))]
    machine_ids = [f"{get_target_regime(cfg)}::{idx}" for idx in range(len(machine_entries))]

    detector_dataset = Dataset.from_dict(
        {
            "id": human_ids + machine_ids,
            "text": human_entries + machine_entries,
            "label": [0] * len(human_entries) + [1] * len(machine_entries),
        }
    )

    return (
        detector_dataset,
        sampled_dataset_path,
        machine_texts_path,
        len(human_entries),
        len(machine_entries),
    )


def print_info(cfg: DictConfig) -> None:
    log.info("Running detector evaluation...")
    log.info(f"Detector:        {get_detector_name(cfg)}")
    log.info(f"Dataset:         {cfg.dataset.name}")
    log.info(f"Generator:       {cfg.model.name}")
    log.info(f"Target regime:   {get_target_regime(cfg)}")
    log.info(f"Sampled dataset: {cfg.experiment.paths.sampled_file}")
    log.info(f"Machine texts:   {cfg.experiment.paths.machine_file}")
    log.info(f"Metrics file:    {cfg.experiment.paths.metrics_file}")
    log.info(f"Temperature:     {cfg.experiment.generation.temperature}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    validate(cfg)
    print_info(cfg)

    (
        detector_dataset,
        sampled_dataset_path,
        machine_texts_path,
        human_count,
        machine_count,
    ) = build_detector_dataset(cfg)

    metrics_file = Path(cfg.experiment.paths.metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if metrics_file.exists():
        print(f"Metrics file already exists: {metrics_file}")
        sys.exit(0)


    detector = load_detector_model(cfg)
    predictions = detector.predict(detector_dataset)
    target_regime = get_target_regime(cfg)
    metrics = Metrics(
        predictions,
        tpr_fpr_target=get_tpr_fpr_target(cfg),
        comparison_name=f"{target_regime}_vs_human",
        metadata={
            "detector_name": get_detector_name(cfg),
            "dataset_name": cfg.dataset.name,
            "generator_name": cfg.model.name,
            "target_regime": target_regime,
            "machine_postfix": cfg.get("machine_postfix"),
            "sampled_dataset_path": str(sampled_dataset_path),
            "machine_texts_path": str(machine_texts_path),
            "rewriter_name": (
                cfg.model.name if target_regime in {"h2l", "llm2l"} else None
            ),
            "no_human": human_count,
            "no_machine": machine_count,
        },
    )

    metrics_file = Path(cfg.experiment.paths.metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Detector:        {get_detector_name(cfg)}")
    log.info(f"Sampled dataset: {sampled_dataset_path}")
    log.info(f"Machine texts:   {machine_texts_path}")
    log.info(f"Human entries:   {human_count}")
    log.info(f"Machine entries: {machine_count}")
    log.info(f"Combined rows:   {len(detector_dataset)}")
    log.info(metrics)

    metrics.save_to_file(
        metrics_file,
        get_detector_name(cfg),
        cfg.dataset.name,
        cfg.model.name,
        cfg.experiment.generation.temperature,
    )


if __name__ == "__main__":
    main()
