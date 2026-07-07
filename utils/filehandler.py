import os
from typing import TypedDict


def create_file_if_not_exists(file_path):
    """
    Creates a file if it does not already exist.
    
    Args:
        file_path (str): The path to the file to create.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('')  # Create an empty file
        print(f"Created file: {file_path}")
    else:
        print(f"File already exists: {file_path}")


class FileHandler:
    """This classes handles file operations such as creating, appending, and reading files.
     It is used for storing generated texts from generative models. 
     The dataset is sent as Dataset HuggingFace object.
    """
    def __init__(self, filepath, mode='r'):
        self.filepath = filepath
        # self.filename = os.path.join('datasets', f'{prefix + "_" if prefix != "" else ""}{"human_" if is_human else ""}{model}_{dataset_name}.txt')
        self.open_file(mode)  # Open the file for appending
        # self.texts_to_generate = dataset['prompt']

        # self.open_file('a')  # Open the file for appending

    @staticmethod
    def exists(filepath):
        """
        Checks if a file exists at the given filepath.
        
        Args:
            filepath (str): The path to the file to check.
        """
        return os.path.exists(filepath)

    def open_file(self, mode='a'):
        """
        Opens the file for appending or reading.
        
        Args:
            mode (str): The mode in which to open the file ('a' for append, 'r' for read).
        
        Returns:
            file object: The opened file object.
        """
        create_file_if_not_exists(self.filepath)
        self.file = open(self.filepath, mode, encoding='utf-8')
        print(f"Opened file: {self.filepath} in mode: {mode}")

    def append_text(self, text):
        """
        Appends a text to the opened file.
        
        Args:
            text (str): The text to append to the file.
        """
        self.file.write(text + '\n')

    def close_file(self):
        """
        Closes the opened file.
        """
        if self.file:
            self.file.close()
            print(f"Closed file: {self.filepath}")

    def get_file_length(self):
        """
        Returns the current line number in the file.
        
        Returns:
            int: The number of lines in the file.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            length = sum(1 for _ in f)
        print(f"File length: {length}")
        return length

    def get_lines(self):
        """
        Returns all lines in the file as a list.
        
        Returns:
            list: A list of lines in the file.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Retrieved {len(lines)} lines from file: {self.filepath}")
        return lines

    def write_line(self, line):
        """
        Writes a single line to the file.
        
        Args:
            line (str): The line to write to the file.
        """
        self.file.write(line + '\n')


    # def get_texts_to_paraphrase(self, index = -1):
    #     """
    #     Returns the texts to paraphrase
        
    #     Args:
    #         index (int): The index of the text to retrieve.
        
    #     Returns:
    #         str: The original text at the specified index.
    #     """
    #     if index == -1:
    #         index = self.get_current_line()
    #     return self.texts_to_generate[index:]


class FileHandlerPilot(FileHandler):
    def __init__(self, filepath):
        super().__init__(filepath)


def get_name_for_metrics(filepath: str) -> str:
    """
    Given a filepath, returns a name suitable for metrics files.
    
    Args:
        filepath (str): The path to the original file.

    Returns:
        str: A name suitable for metrics files.
    """
    return filepath.replace('.txt', constants.FILE_METRICS_POSTFIX)

def get_name_for_cleaned_metrics(filepath: str) -> str:
    """
    Given a filepath, returns a name suitable for metrics files.
    
    Args:
        filepath (str): The path to the original file.

    Returns:
        str: A name suitable for metrics files.
    """
    return filepath.replace('.txt', constants.CLEANED_METRICS)


class GenerationInformation(TypedDict):
    generator: str
    dataset_key: str
    temperature: float
    is_pilot: bool

def get_from_name_information(filepath: str) -> GenerationInformation:
    """
    Given a filepath, extracts the generator model, dataset key, and temperature.

    Args:
        filepath (str): The path to the original file.

    Returns:
        GenerationInformation: A dictionary containing the extracted information.
    """
    # Example: data/xsum_llama-3b_0.1_pilot.txt
    parts = filepath.split('/')[-1].replace('.txt', '').split('_')
    return GenerationInformation(
        generator=parts[1],
        dataset_key=parts[0],
        temperature=float(parts[2]),
        is_pilot=parts[3] == "pilot"
    )