import unittest
from benchmarking.benchmark import *


class TestBenchmark(unittest.TestCase):

    def test_simple_addition(self):
        script_code = "x = 10\ny = 20\nz = x + y"
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_empty_script(self):
        script_code = ""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_large_loop(self):
        script_code = """
for i in range(1000000):
    pass
"""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_function_execution(self):
        script_code = """
def test_function():
    return sum(range(1000))

result = test_function()
"""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_simple_addition(self):
        script_code = "x = 10\ny = 20\nz = x + y"
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_empty_script(self):
        script_code = ""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_large_loop(self):
        script_code = """
for i in range(1000000):
    pass
"""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_function_execution(self):
        script_code = """
def test_function():
    return sum(range(1000))

result = test_function()
"""
        exec_time = measure_execution_time(script_code)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)

    def test_simple_addition(self):
        script_code = "x = 10\ny = 20\nz = x + y"
        mem_usage = measure_memory_usage(script_code)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_empty_script(self):
        script_code = ""
        mem_usage = measure_memory_usage(script_code)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_large_loop(self):
        script_code = """
for i in range(1000000):
    pass
"""
        mem_usage = measure_memory_usage(script_code)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_function_execution(self):
        script_code = """
def test_function():
    return sum(range(1000))

result = test_function()
"""
        mem_usage = measure_memory_usage(script_code)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_simple_addition(self):
        script_code = "x = 10\ny = 20\nz = x + y"
        exec_time, mem_usage = run_benchmark(script_code, bool=False)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_empty_script(self):
        script_code = ""
        exec_time, mem_usage = run_benchmark(script_code, bool=False)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_large_loop(self):
        script_code = """
for i in range(1000000):
    pass
"""
        exec_time, mem_usage = run_benchmark(script_code, bool=False)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_function_execution(self):
        script_code = """
def test_function():
    return sum(range(1000))

result = test_function()
"""
        exec_time, mem_usage = run_benchmark(script_code, bool=False)
        self.assertIsInstance(exec_time, float)
        self.assertGreaterEqual(exec_time, 0)
        self.assertIsInstance(mem_usage, list)
        self.assertGreaterEqual(len(mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in mem_usage))

    def test_comparison(self):
        original_script_code = "x = 10\ny = 20\nz = x + y"
        optimised_script_code = "x = 10\ny = 20\nz = 30"
        (
            original_exec_time,
            original_mem_usage,
            optimised_exec_time,
            optimised_mem_usage,
        ) = run_comparison_benchmark(
            original_script_code, optimised_script_code, bool=False
        )
        self.assertIsInstance(original_exec_time, float)
        self.assertGreaterEqual(original_exec_time, 0)
        self.assertIsInstance(original_mem_usage, list)
        self.assertGreaterEqual(len(original_mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in original_mem_usage))
        self.assertIsInstance(optimised_exec_time, float)
        self.assertGreaterEqual(optimised_exec_time, 0)
        self.assertIsInstance(optimised_mem_usage, list)
        self.assertGreaterEqual(len(optimised_mem_usage), 1)
        self.assertTrue(all(isinstance(mem, float) for mem in optimised_mem_usage))


if __name__ == "__main__":
    unittest.main()
