import time
from memory_profiler import memory_usage


def measure_execution_time(script_code):
    """
    Measures the execution time of a given script.
    Args:
        script_code (str): The code of the script to be executed as a string.
    Returns:
        float: The execution time of the script in seconds.
    """
    start_time = time.perf_counter()
    exec(script_code)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return execution_time


def measure_memory_usage(script_code):
    """
    Measures the memory usage of a given script.

    Args:
        script_code (str): The code of the script to be executed and measured.

    Returns:
        list: A list of memory usage measurements in MB.
    """
    mem_usage = memory_usage((exec, (script_code,)))
    return mem_usage


def run_benchmark(script_code, bool=True):
    """
    Runs a benchmark on the provided script code to measure its execution time and memory usage.

    Parameters:
        script_code (str): The code to be benchmarked.
        bool (bool): If True, prints the benchmark results. Default is True.

    Returns:
        tuple: A tuple containing the execution time (float) in seconds and memory usage (list of floats) in MiB.
    """
    exec_time = measure_execution_time(script_code)
    mem_usage = measure_memory_usage(script_code)

    if bool:
        print("Benchmark Results:")
        print("Execution Time: {:.8f} seconds".format(exec_time))
        print("Memory Usage: {:.4f} MiB".format(max(mem_usage)))

    return exec_time, mem_usage


def run_comparison_benchmark(original_script_code, optimised_script_code, bool=True):
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
    # Measure execution time
    original_exec_time = measure_execution_time(original_script_code)
    optimised_exec_time = measure_execution_time(optimised_script_code)

    # Measure memory usage
    original_mem_usage = measure_memory_usage(original_script_code)
    optimised_mem_usage = measure_memory_usage(optimised_script_code)

    if bool:
        print("Benchmark Results:")
        print("Execution Time (Original): {:.8f} seconds".format(original_exec_time))
        print("Execution Time (Optimised): {:.8f} seconds".format(optimised_exec_time))
        print("Memory Usage (Original): {:.2f} MiB".format(max(original_mem_usage)))
        print("Memory Usage (Optimised): {:.2f} MiB".format(max(optimised_mem_usage)))

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
print(x + y)
"""
    optimised_script_code = """
x = 10
y = 20
print(30)
"""

    run_benchmark(optimised_script_code)
    run_comparison_benchmark(original_script_code, optimised_script_code)
