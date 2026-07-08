from pathlib import Path

from datasets import load_from_disk
from omegaconf import DictConfig
import hydra

from utils.dataset import combine_human_ai_texts
from utils.detector import load_detector_model
from utils.evaluation.Metrics import Metrics


REPO_ROOT = Path(__file__).resolve().parent


def get_sampled_dataset_path(cfg: DictConfig) -> Path:
    custom_path = cfg.get("sampled_dataset_path")
    if custom_path:
        return Path(custom_path)
    return REPO_ROOT / "output" / f"{cfg.dataset.name}_sampled"


def get_machine_texts_path(cfg: DictConfig) -> Path:
    custom_path = cfg.get("machine_texts_path")
    if custom_path:
        return Path(custom_path)
    return REPO_ROOT / "output" / f"{cfg.dataset.name}_{cfg.model.name}_h2l.txt"


def load_machine_entries(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def build_detector_dataset(cfg: DictConfig):
    sampled_dataset_path = get_sampled_dataset_path(cfg)
    machine_texts_path = get_machine_texts_path(cfg)

    if not sampled_dataset_path.exists():
        raise FileNotFoundError(f"Sampled dataset not found: {sampled_dataset_path}")
    if not machine_texts_path.exists():
        raise FileNotFoundError(f"Machine texts file not found: {machine_texts_path}")

    sampled = load_from_disk(str(sampled_dataset_path))
    human_entries = list(sampled["text"])
    machine_entries = load_machine_entries(machine_texts_path)
    detector_dataset = combine_human_ai_texts(human_entries, machine_entries)

    return detector_dataset, sampled_dataset_path, machine_texts_path, len(human_entries), len(machine_entries)


def get_detector_name(cfg: DictConfig) -> str:
    if "model_name" in cfg.detector:
        return cfg.detector.model_name
    if "observer_name_or_path" in cfg.detector:
        return "binoculars"
    return cfg.detector.detector_type


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    detector_dataset, sampled_dataset_path, machine_texts_path, human_count, machine_count = build_detector_dataset(cfg)
    detector = load_detector_model(cfg)
    predictions = detector.predict(detector_dataset)
    metrics = Metrics(predictions)

    print(f"Detector:        {get_detector_name(cfg)}")
    print(f"Sampled dataset: {sampled_dataset_path}")
    print(f"Machine texts:   {machine_texts_path}")
    print(f"Human entries:   {human_count}")
    print(f"Machine entries: {machine_count}")
    print(f"Combined rows:   {len(detector_dataset)}")
    print(metrics)


if __name__ == "__main__":
    main()
