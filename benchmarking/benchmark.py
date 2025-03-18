import time
import random

def measure_execution_time(script_code):
    start_time = time.perf_counter()
    exec(script_code)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    return execution_time

def run_full_benchmark(original_script_code, optimised_script_code):
    # Measure execution time for original script
    original_time = measure_execution_time(original_script_code)
    print(f"Original Script Execution Time: {original_time:.8f} seconds")
    
    # Measure execution time for optimized script
    optimised_time = measure_execution_time(optimised_script_code)
    print(f"Optimised Script Execution Time: {optimised_time:.8f} seconds")

# Example test case in code as a string
original_script_code = """
def slow_task(size=1000):
    total = 0
    for i in range(size):
        for j in range(size):
            total += (i * j) ** 0.5  # Some basic but slow computation
    return total

def generate_data(size=1000):
    return [random.randint(1, 100) for _ in range(size)]

def process_data(data):
    # Inefficient sorting: sorting before summing every time
    data.sort()
    return sum(data)

def main():
    # Simulate data generation and processing
    data = generate_data(size=10000)  # Generate a dataset of 10,000 items
    processed_data = process_data(data)

    # Run a slow computational task
    result = slow_task(size=100)

    return processed_data, result
"""

optimised_script_code = """
def optimized_slow_task(size=1000):

    total = 0
    # Use a more efficient loop and pre-compute the power of values
    for i in range(size):
        for j in range(size):
            total += (i * j) ** 0.5  # Same basic computation but simplified
    return total

def generate_data(size=1000):
    return [random.randint(1, 100) for _ in range(size)]

def optimized_process_data(data):
    # Directly sum the data, skipping unnecessary sorting
    return sum(data)

def main():
    # Simulate data generation and processing
    data = generate_data(size=10000)  # Generate a dataset of 10,000 items
    processed_data = optimized_process_data(data)

    # Run the optimized computational task
    result = optimized_slow_task(size=100)

    return processed_data, result
"""


run_full_benchmark(original_script_code, optimised_script_code)