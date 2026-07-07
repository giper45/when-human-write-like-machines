import os
from datasets import load_from_disk
from utils.logger import log

class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None

    def load(self, dataset_name: str):
        full_path = os.path.join(self.dataset_path, dataset_name)
        log.info(f"Loading dataset from {full_path}")
        self.dataset = load_from_disk(full_path)

    def get_first_n_samples(self, n: int): 
        return self.dataset.select(range(n))

    def save(self, dataset, dataset_name: str):
        output_path = os.path.join(self.dataset_path, dataset_name)
        log.info(f"Saving dataset to {output_path}")
        dataset.save_to_disk(output_path)

    




# def get_local_dataset_path(dataset_name = ""):
#     # Return homedir + "/data/datasets/" + self.dataset_name
#     if dataset_name == "":
#         return os.path.join(constants.DATASET_PATH)
#     return os.path.join(constants.DATASET_PATH, dataset_name)


# def get_notoxic_dataset(dataset_name):
#     return load_from_disk(get_local_dataset_path(dataset_name) + "_notoxic")

# def save_final_dataset(dataset, cfg):
