import json
import os

"""
This module provides utility functions for loading JSON and text files.

Functions:
    load_json_file(file_path: str) -> dict or None:
        Loads a JSON file from the specified file path.
        Returns the parsed JSON data as a dictionary, or None if an error occurs.
        
    load_text_file(file_path: str) -> str or None:
        Loads a text file from the specified file path.
        Returns the file content as a string, or None if an error occurs.
        
    load_json_files_from_directory(directory_path: str, file_names: list) -> dict:
        Loads multiple JSON files from the specified directory.
        Returns a dictionary where keys are file names and values are the parsed JSON data.
        
    load_text_files_from_directory(directory_path: str, file_names: list) -> dict:
        Loads multiple text files from the specified directory.
        Returns a dictionary where keys are file names and values are the file contents.
"""


def load_json_file(file_path: str):
    """
    Loads a JSON file from the specified file path.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:

        dict: The parsed JSON data as a dictionary.

        None: If there was an error loading the file.

    Raises:
        json.JSONDecodeError: If the file contains invalid JSON.
        OSError: If there is an issue with file access.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Error loading JSON file {file_path} -> {e}")
        return None


def load_text_file(file_path: str):
    """
    Loads the content of a text file.

    Args:
        file_path (str): The path to the text file to be loaded.

    Returns:
        str: The content of the text file if successfully read.

        None: If there is an error reading the file.

    Raises:
        OSError: If there is an issue with file access.
        Exception: For any other exceptions that may occur.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except (OSError, Exception) as e:
        print(f"Warning: Error reading text file {file_path} -> {e}")
        return None


def load_json_files_from_directory(directory_path: str):
    """
    Load JSON files from a specified directory.

    This function scans the given directory for JSON files, loads their content,
    and returns a dictionary where the keys are the file names and the values are
    the loaded JSON data. If a file is missing or empty, a warning message is printed.

    Args:
        directory_path (str): The path to the directory containing JSON files.

    Returns:
        dict: A dictionary containing the loaded JSON data, with file names as keys.
    """

    data = {}
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                data[file_name] = load_json_file(file_path)
            else:
                print(
                    f"Warning: {file_name} is missing or empty in {directory_path}")
    return data


def load_text_files_from_directory(directory_path: str):
    """
    Loads text files from a specified directory.

    This function scans the given directory for text files, loads their content,
    and returns a dictionary where the keys are the file names and the values are
    the loaded text data. If a file is missing or empty, a warning message is printed.

    Args:
        directory_path (str): The path to the directory containing the text files.
        file_names (list): A list of file names to be loaded from the directory.

    Returns:
        dict: A dictionary where the keys are file names and the values are the contents of the text files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If the specified path is not a directory.

    Notes:
        - If a file is missing or empty, a warning message will be printed.
        - Only files that exist and are non-empty will be loaded.
    """

    data = {}
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                data[file_name] = load_text_file(file_path)
            else:
                print(
                    f"Warning: {file_name} is missing or empty in {directory_path}")
    return data
