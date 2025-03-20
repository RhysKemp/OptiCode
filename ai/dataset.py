import json
import os
import torch
from tqdm import tqdm
from engine.ast_parser import ASTParser
from torch.utils.data import Dataset
from utils.file_loader import *

# TODO: figure out how to clean the data properly - possibly handling it better in file_loader.py


class CustomDataset(Dataset):
    """
    A custom dataset class for loading and processing code datasets.
    Attributes:
        dataset_dir (str): The directory containing the dataset.
        parser (ASTParser): An instance of the ASTParser class.
        data (dict): A dictionary containing the loaded dataset.
        processed_data (list): A list of tuples containing processed ASTs.
    Methods:
        load_dataset():
            Loads the dataset from the specified directory.
            Returns:
                dict: A dictionary containing the loaded dataset.
        process_dataset():
            Processes the loaded dataset to generate ASTs.
            Returns:
                list: A list of tuples containing processed ASTs.
        __len__():
            Returns the number of processed data items.
            Returns:
                int: The number of processed data items.
        __getitem__(index):
            Retrieves the processed data item at the specified index.
            Args:
                index (int): The index of the data item to retrieve.
            Returns:
                tuple: A tuple containing the inefficient and accepted ASTs.
    """

    def __init__(self, dataset_dir: str):
        """
        Initialises the Dataset object.

        Args:
            dataset_dir (str): The directory where the dataset is stored.

        Attributes:
            dataset_dir (str): The directory where the dataset is stored.
            parser (ASTParser): An instance of the ASTParser class.
            data (Any): The loaded dataset.
            processed_data (Any): The processed dataset.
        """
        self.dataset_dir = dataset_dir
        self.parser = ASTParser()
        self.data = self.load_dataset()
        self.processed_data = self.process_dataset()

    def load_dataset(self):
        """
        Loads datasets from the specified directory.

        This method scans through the dataset directory and loads data from
        subdirectories that contain the required files: "input_output.json",
        "Accepted.json", and a directory named "Acc_tle_solutions". It ensures
        that these files are non-empty before loading their contents.

        Returns:
            dict: A dictionary where each key is the path to a valid dataset
            directory, and each value is another dictionary containing:
                - "input_output": Data loaded from "input_output.json".
                - "accepted": Data loaded from "Accepted.json".
                - "acc_tle_solutions": List of data loaded from files in
                  "Acc_tle_solutions" directory.
        """
        data = {}
        for entry in tqdm(os.scandir(self.dataset_dir), desc="Loading directories", unit="dir"):
            if entry.is_dir():
                root = entry.path
                files = os.listdir(root)

                input_output_data = None
                accepted_data = None
                acc_tle_solutions = []

                # Load input_output.json if exists and non-empty
                input_output_path = os.path.join(root, "input_output.json")
                if (
                    "input_output.json" in files
                    and os.path.getsize(input_output_path) > 0
                ):
                    input_output_data = load_json_file(input_output_path)

                # Load Accepted.json if exists and non-empty
                accepted_path = os.path.join(root, "Accepted.json")
                if "Accepted.json" in files and os.path.getsize(accepted_path) > 0:
                    accepted_data = load_text_file(accepted_path)

                # inefficient solutions
                acc_tle_dir = os.path.join(root, "Acc_tle_solutions")
                acc_tle_solutions = load_text_files_from_directory(acc_tle_dir)

                # Ensure valid data
                if input_output_data and accepted_data and acc_tle_solutions:

                    data[root] = {
                        "input_output": input_output_data,
                        "accepted": accepted_data,
                        "acc_tle_solutions": acc_tle_solutions,
                    }
                else:
                    print(f"Skipping incomplete dataset at {root}")

        return data

    def process_dataset(self):
        """
        Processes the dataset by parsing the abstract syntax trees (AST) of both
        inefficient and efficient code solutions.

        For each directory in the dataset, it retrieves the input/output data,
        accepted code, and inefficient codes. It then parses the AST for each
        inefficient code and the accepted code, and appends a tuple of these ASTs
        to the processed list.
        Returns:
            list: A list of tuples where each tuple contains the AST of an
                  inefficient code and the AST of the corresponding accepted code.
        """
        processed = []
        for dir in tqdm(self.data.values(), desc="Processing directories", unit="dir"):
            input_output = dir["input_output"]
            efficient_code = dir["accepted"]
            inefficient_codes = dir["acc_tle_solutions"]

            for inefficient_code in inefficient_codes.values():
                try:
                    self.parser.parse_ast(inefficient_code)
                    inefficient_ast = self.parser.tree
                    self.parser.parse_ast(efficient_code)
                    efficient_ast = self.parser.tree

                    processed.append((inefficient_ast, efficient_ast))
                except SyntaxError as e:
                    print(f"SyntaxError: {e}")
                    print(f"Source code: {repr(inefficient_code)}")
                    continue

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        return self.processed_data[index]
