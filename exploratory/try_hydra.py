from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def save_resolved_config(cfg: DictConfig):
    path = Path(cfg.paths.resolved_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path


def validate_backends(cfg: DictConfig):
    for model in cfg.models:
        if model.backend != cfg.generation.backend:
            raise ValueError(
                f"Backend mismatch: generation.backend={cfg.generation.backend}, "
                f"but model {model.name} has backend={model.backend}"
            )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    validate_backends(cfg)

    print("\n=== EXPERIMENT ===")
    print(f"name: {cfg.experiment.name}")
    print(f"stage: {cfg.stage}")

    print("\n=== GENERATION ===")
    print(f"backend: {cfg.generation.backend}")
    print(f"temperature: {cfg.generation.temperature}")
    print(f"top_p: {cfg.generation.top_p}")
    print(f"top_k: {cfg.generation.top_k}")

    print("\n=== MODELS ===")
    print(f"model_set: {cfg.model_set.name}")
    for model in cfg.models:
        print(
            f"- {model.name}: "
            f"{model.model_id}, "
            f"dtype={model.dtype}, "
            f"quantization={model.quantization}"
        )

    print("\n=== DATASETS ===")
    for ds in cfg.dataset.sources:
        print(f"- {ds.name}: {ds.n_samples}")

    print("\n=== REGIMES ===")
    print(list(cfg.regimes))

    saved = save_resolved_config(cfg)
    print(f"\nSaved resolved config: {saved}")

    print("\n=== FULL CONFIG ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()