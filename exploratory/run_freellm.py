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
from prompts import freellm, h2l, llm2lm, system
from utils.transformer import load_model_bf16, generate_text

# with initialize(version_base=None, config_path="../conf"):
#     cfg = compose(config_name="config")

def clean_up(file_handler: FileHandler):
    """
    Function to be called upon script exit to perform cleanup tasks.
    """
    print("Cleaning up resources...")
    clear_gpu_full()  # Uncomment if you want to clear GPU memory
    file_handler.close_file()  # Ensure the file is closed
    print("Cleanup complete.")



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    system_prompt = system.get_system_prompt()

    output_name = f"output/{cfg.dataset.name}_{cfg.model.name}_freellm.txt"
    if FileHandler.exists(output_name):
        print(f"Output file {output_name} already exists. Exiting to avoid overwriting.")
        return
    else:
        file_handler = FileHandler(output_name, 'w')
        atexit.register(clean_up, file_handler)  # <--- Registra la funzione di clean-up per essere eseguita alla chiusura dello script
        model_id = cfg.model.model_id
        dataset = load_from_disk(f"output/{cfg.dataset.name}_sampled")
        model, tokenizer = load_model_bf16(model_id)
        range_dataset = add_range_to_dataset(dataset, 
                     cfg.experiment.ranges.short.range, 
                     cfg.experiment.ranges.medium.range, 
                        cfg.experiment.ranges.long.range)



        def get_short_long(cfg, range_val):
          range_res = cfg.experiment.ranges[range_val].range
          return {"short": range_res[0], "long": range_res[1]}

        short_long_dataset = range_dataset.map(lambda x: get_short_long(cfg, x['range'])) 
        short_long_dataset = short_long_dataset.remove_columns(['range'])
        topic_dataset = add_topic_to_dataset(short_long_dataset, cfg.dataset.topic_source)
        


        for d in tqdm(topic_dataset, desc="Processing topics"):
            min_length = d['short']
            max_length = d['long']
            topic = d['topic']
            freellm_prompt = freellm.get_prompt(topic, min_length, max_length)

            print(freellm_prompt)

            llm_text = generate_text(tokenizer, model, freellm_prompt, system_prompt)
            # print(f"Generated text: {llm_text}")
            file_handler.write_line(llm_text)


if __name__ == "__main__":
    main()
