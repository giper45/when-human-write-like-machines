import os
from datasets import concatenate_datasets, load_from_disk
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

LABEL_HUMAN = 0
LABEL_MACHINE = 1
def combine_human_ai_dataset(dataset_human, 
                             dataset_machine, 
                             seed=42):

    # Add labels to human and machine datasets
    dataset_human = dataset_human.map(
        lambda x: {"label": LABEL_HUMAN}
    )
    dataset_machine = dataset_machine.map(
        lambda x: {"label": LABEL_MACHINE}
    )

    # Concatenate datasets
    combined_dataset = concatenate_datasets([dataset_human, dataset_machine])

    # Shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle(seed=seed)

    # Generate new ids after combination and shuffling
    combined_dataset = combined_dataset.map(
        lambda x, idx: {"id": idx},
        with_indices=True
    )

    return combined_dataset

def save_texts_line_by_line(dataset, output_path, text_column="text"):
    with open(output_path, "w", encoding="utf-8") as f:
        for text in dataset[text_column]:
            text = text.replace("\n", " ").strip()
            f.write(text + "\n")