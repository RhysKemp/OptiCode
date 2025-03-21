

import subprocess


def validate_code(code: str, input: str, expected_output: str):
    """
    Validates the given code by running it with the provided input and comparing the output to the expected output.

    Args:
        code (str): The Python code to be executed.
        input (str): The input to be passed to the code.
        expected_output (str): The expected output to be compared against the actual output.

    Returns:
        dict: A dictionary containing:
            - "actual_output" (str): The actual output produced by running the code.
            - "error_output" (str): Any error messages produced during the execution of the code.
            - "success" (bool): True if the actual output matches the expected output, False otherwise.
    """
    result = subprocess.run(
        ["python", "-c", code],
        input=input,
        capture_output=True,
        text=True,
        timeout=10
    )
    actual_output = result.stdout
    error_output = result.stderr

    print("\nActual Output:")
    print(actual_output)

    return {
        "actual_output": actual_output,
        "error_output": error_output,
        "success": actual_output == expected_output.strip()
    }
