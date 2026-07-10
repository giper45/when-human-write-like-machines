import sys
from tqdm import tqdm

sys.path.append('../')
from utils.device import clear_gpu_full
from datasets import load_dataset, load_from_disk
from utils.range_classification import add_range_to_dataset
from utils.topic_selector import add_topic_to_dataset

from hydra import compose, initialize
import hydra

import atexit  # <--- Importi il modulo globale di clean-up
from omegaconf import DictConfig, OmegaConf
from utils.filehandler import FileHandler

# from utils.device import clear_gpu_full
from prompts import llm2lm, system
from utils.transformer import load_model_bf16, generate_text
from utils.logger import log

# with initialize(version_base=None, config_path="../conf"):
#     cfg = compose(config_name="config")

def clean_up(file_handler: FileHandler):
    """
    Function to be called upon script exit to perform cleanup tasks.
    """
    log.info("Cleaning up resources...")
    clear_gpu_full()  # Uncomment if you want to clear GPU memory
    file_handler.close_file()  # Ensure the file is closed
    log.info("Cleanup complete.")



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    system_prompt = system.get_system_prompt()

    freellm_name = f"{cfg.dataset.freellm_name}.txt"
    log.info(f"Reading input from {freellm_name}")
    input_handler = FileHandler(f"output/{freellm_name}", 'r')
    lines = input_handler.get_lines()
    input_handler.close_file()
    log.info(f"Number of lines in input: {len(lines)}")
    output_name = f"output/{cfg.dataset.name}_{cfg.model.name}_llm2l.txt"
    if FileHandler.exists(output_name):
        log.info(f"Output file {output_name} already exists. Exiting to avoid overwriting.")
        return
    else:
        file_handler = FileHandler(output_name, 'w')
        atexit.register(clean_up, file_handler)  # <--- Registra la funzione di clean-up per essere eseguita alla chiusura dello script
        model_id = cfg.model.model_id
        model, tokenizer = load_model_bf16(model_id)

        for free_llm_text in tqdm(lines, desc="Processing topics"):
            llm2lm_prompt = llm2lm.get_prompt(free_llm_text)

            log.info(f"Processing: {llm2lm_prompt}")

            llm2lm_text = generate_text(tokenizer, model, llm2lm_prompt, system_prompt)
            file_handler.write_line(llm2lm_text)


if __name__ == "__main__":
    main()

