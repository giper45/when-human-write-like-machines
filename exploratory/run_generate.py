import sys
from tqdm import tqdm
sys.path.append('../')

from hydra import compose, initialize
import hydra

import atexit  # <--- Importi il modulo globale di clean-up
from omegaconf import DictConfig, OmegaConf
from utils.filehandler import FileHandler

# from utils.device import clear_gpu_full
from prompts import freellm, h2l, llm2lm, system
from utils.transformer import load_model_bf16, generate_text

# with initialize(version_base=None, config_path="../conf"):
#     cfg = compose(config_name="config")

def clean_up():
    """
    Function to be called upon script exit to perform cleanup tasks.
    """
    # print("Cleaning up resources...")
    # # clear_gpu_full()  # Uncomment if you want to clear GPU memory
    # file_handler.close_file()  # Ensure the file is closed
    # file_handler_write.close_file()  # Ensure the output file is closed
    # print("Cleanup complete.")



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    atexit.register(clean_up)  # <--- Registra la funzione di clean-up per essere eseguita alla chiusura dello script
    system_prompt = system.get_system_prompt()

    output_name = f"output/{cfg.dataset.name}_{cfg.model.name}_h2l.txt"
    if FileHandler.exists(output_name):
        print(f"Output file {output_name} already exists. Exiting to avoid overwriting.")
        return
    else:
        model_id = cfg.model.model_id
        file_handler = FileHandler(f"output/{cfg.dataset.name}_sampled.txt")
        file_handler_write = FileHandler(output_name, mode='w')

        lines = file_handler.get_lines()
        model, tokenizer = load_model_bf16(model_id)






        for t in tqdm(lines, desc="Processing lines"):
            h2l_prompt = h2l.get_prompt(t)
            llm_text = generate_text(tokenizer, model, h2l_prompt, system_prompt)
            # print(f"Generated text: {llm_text}")
            file_handler_write.write_line(llm_text)


if __name__ == "__main__":
    main()

