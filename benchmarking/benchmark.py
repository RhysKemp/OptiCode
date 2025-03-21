import subprocess
import tempfile
import time
from tqdm import tqdm
from memory_profiler import memory_usage


def measure_execution_time(source_code, inputs=None, timeout: int = 30):
    """
    Measures the execution time of a given script.
    Args:
        script_code (str): The code of the script to be executed as a string.
        inputs (list of str): The input data to be provided to the script.
        timeout (int): The maximum time to wait for the script to complete.
    Returns:
        float: The execution time of the script in seconds.
    """
    if inputs is None:
        inputs = [""]

    total_execution_time = 0

    for input_data in inputs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(source_code.encode())
            temp_file.flush()
            temp_file_name = temp_file.name

        start_time = time.perf_counter()
        try:
            subprocess.run(
                ["python", temp_file_name],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            print(f"Error: Script timed out after {timeout} seconds")
        except Exception as e:
            print(f"Error during execution: {e}")
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        total_execution_time += execution_time

    return total_execution_time


def measure_memory_usage(source_code, inputs=None):
    """
    Measures the memory usage of a given script.

    Args:
        script_code (str): The code of the script to be executed and measured.
        inputs (list of str): The input data to be provided to the script.

    Returns:
        list: A list of memory usage measurements in MB.
    """
    if inputs is None:
        inputs = [""]

    total_mem_usage = []

    for input_data in inputs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(source_code.encode())
            temp_file.flush()
            temp_file_name = temp_file.name

        try:
            mem_usage = memory_usage((subprocess.run, (["python", temp_file_name],), {
                                     "input": input_data, "capture_output": True, "text": True}))
            total_mem_usage.extend(mem_usage)
        except Exception as e:
            print(f"Error during memory measurement: {e}")

    return total_mem_usage


def run_benchmark(script_code, display=False):
    """
    Runs a benchmark on the provided script code to measure its execution time and memory usage.

    Parameters:
        script_code (str): The code to be benchmarked.
        bool (bool): If True, prints the benchmark results. Default is True.

    Returns:
        tuple: A tuple containing the execution time (float) in seconds and memory usage (list of floats) in MiB.
    """
    with tqdm(total=2, desc="Benchmarking", unit="step") as pbar:
        exec_time = measure_execution_time(script_code)
        pbar.update(1)
        mem_usage = measure_memory_usage(script_code)
        pbar.update(1)

    if display:
        print("Benchmark Results:")
        print("Execution Time: {:.8f} seconds".format(exec_time))
        print("Memory Usage: {:.4f} MiB".format(max(mem_usage)))

    return exec_time, mem_usage


def run_comparison_benchmark(original_script_code, optimised_script_code, display=False, input: str = None):
    """
    Runs a benchmark comparison between the original and optimised script code.

    This function measures and compares the execution time and memory usage of the
    original and optimised script code. Optionally, it prints the benchmark results.

    Parameters:
        original_script_code (str): The code of the original script to be benchmarked.
        optimised_script_code (str): The code of the optimised script to be benchmarked.
        bool (bool): If True, prints the benchmark results. Default is True.

    Returns:
        tuple: A tuple containing:
            - original_exec_time (float): Execution time of the original script in seconds.
            - original_mem_usage (list): Memory usage of the original script in MiB.
            - optimised_exec_time (float): Execution time of the optimised script in seconds.
            - optimised_mem_usage (list): Memory usage of the optimised script in MiB.
    """
    with tqdm(total=4, desc="Comparison Benchmarking", unit="step") as pbar:
        # Measure execution time
        original_exec_time = measure_execution_time(
            original_script_code, input)
        pbar.update(1)
        optimised_exec_time = measure_execution_time(
            optimised_script_code, input)
        pbar.update(1)

        # Measure memory usage
        original_mem_usage = measure_memory_usage(original_script_code)
        pbar.update(1)
        optimised_mem_usage = measure_memory_usage(optimised_script_code)
        pbar.update(1)

    if display:
        print("Benchmark Results:")
        print("Execution Time (Original): {:.8f} seconds".format(
            original_exec_time))
        print("Execution Time (Optimised): {:.8f} seconds".format(
            optimised_exec_time))
        print("Memory Usage (Original): {:.2f} MiB".format(
            max(original_mem_usage)))
        print("Memory Usage (Optimised): {:.2f} MiB".format(
            max(optimised_mem_usage)))

    return (
        original_exec_time,
        original_mem_usage,
        optimised_exec_time,
        optimised_mem_usage,
    )


if __name__ == "__main__":
    original_script_code = """
x = 10
y = 20
z = 20 + 45
print(x + y)
"""
    optimised_script_code = """
x = 10
y = 20
print(30)
"""

    run_benchmark(optimised_script_code)
    run_comparison_benchmark(original_script_code, optimised_script_code)
