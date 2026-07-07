from omegaconf import DictConfig, OmegaConf
import hydra




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.experiment.is_pilot)
    # print(cfg.experiment.pilot_length)

if __name__ == "__main__":
    main()

