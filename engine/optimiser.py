import os
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from utils.file_loader import *
from engine.ast_parser import ASTParser
from engine.rule_based_optimisations import RuleBasedOptimisations
from benchmarking.benchmark import run_comparison_benchmark
from benchmarking.validation import validate_code


class CodeOptimiser:
    def __init__(self, model_dir):
        """
        Initialises the optimiser with a given model directory.

        Args:
            model_dir (str): The directory where the model is stored.

        Attributes:
            model_available (bool): Indicates whether the model is available in the specified directory.
            tokeniser (RobertaTokenizer): The tokenizer for the model, initialised if the model is available.
            model (T5ForConditionalGeneration): The model for conditional generation, initialised if the model is available.
            ast_parser (ASTParser): The AST parser, initialised if the model is not available.
            rule_based_optimisations (RuleBasedOptimisations): The rule-based optimisations, initialised if the model is not available.

        If the model is not found in the specified directory, it falls back to AST parsing and rule-based optimisation.
        """
        self.model_available = os.path.exists(model_dir)
        if self.model_available:
            self.tokeniser = RobertaTokenizer.from_pretrained(model_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.model.eval()
        else:
            print(
                "Model not found. Falling back to AST parsing and rule-based optimisation.")
            self.ast_parser = ASTParser()
            self.rule_based_optimisations = RuleBasedOptimisations(
                self.ast_parser)

    def benchmark_code(self, original_code, optimised_code, input: str = None):
        """
        Benchmarks the execution time and memory usage of the original and optimised code.

        Args:
            original_code (str): The original code to benchmark.
            optimised_code (str): The optimised code to benchmark.
            input (str, optional): The input to be used for benchmarking. Defaults to None.

        Returns:
            dict: A dictionary containing the execution time and memory usage for both the original and optimised code.
                - "original_exec_time" (float): Execution time of the original code.
                - "original_mem_usage" (float): Memory usage of the original code.
                - "optimised_exec_time" (float): Execution time of the optimised code.
                - "optimised_mem_usage" (float): Memory usage of the optimised code.
        """
        original_exec_time, original_mem_usage, optimised_exec_time, optimised_mem_usage = run_comparison_benchmark(
            original_code, optimised_code, input)
        return {
            "original_exec_time": original_exec_time,
            "original_mem_usage": original_mem_usage,
            "optimised_exec_time": optimised_exec_time,
            "optimised_mem_usage": optimised_mem_usage
        }

    def validate_output(self, generated_code, input_data, expected_output):
        """
        Validates the generated code by comparing its output with the expected output.

        Args:
            generated_code (str): The code that has been generated and needs to be validated.
            input_data (Any): The input data to be used for running the generated code.
            expected_output (Any): The expected output to compare against the generated code's output.

        Returns:
            bool: True if the generated code's output matches the expected output, False otherwise.
        """
        validation_result = validate_code(
            generated_code, input_data, expected_output)
        return validation_result["success"]

    def optimise_code(self, code, input_data, expected_output):
        """
        Optimises the given code based on the availability of a model.

        If a model is available, it uses the model to optimise the code.
        Otherwise, it uses Abstract Syntax Tree (AST) based optimisation.

        Args:
            code (str): The source code to be optimised.
            input_data (Any): The input data to be used for optimisation.
            expected_output (Any): The expected output to validate the optimisation.

        Returns:
            str: The optimised source code.
        """
        if self.model_available:
            return self.optimise_with_model(code, input_data, expected_output)
        else:
            return self.optimise_with_ast(code, input_data, expected_output)

    def optimise_with_model(self, code, input_data, expected_output):
        """
        Optimises the given code using a pre-trained model and benchmarks its performance.

        Args:
            code (str): The source code to be optimised.
            input_data (Any): The input data to be used for benchmarking and validation.
            expected_output (Any): The expected output to validate the optimised code against.

        Returns:
            Tuple[Optional[str], Any]: A tuple containing the optimised code (if valid) and the benchmarks.
                                       If the optimised code is not valid, returns None for the code.
        """
        tokenised_code = self.tokeniser.encode(code, return_tensors="pt")

        # Generate the optimised code
        with torch.no_grad():
            output = self.model.generate(
                tokenised_code, max_length=512, num_beams=4, early_stopping=True)
        optimised_code = self.tokeniser.decode(
            output[0], skip_special_tokens=True)

        # Benchmark
        benchmarks = self.benchmark_code(code, optimised_code, input_data)

        # Validate
        is_valid = self.validate_output(
            optimised_code, input_data, expected_output)

        if is_valid:
            return optimised_code, benchmarks
        else:
            return None, benchmarks

    def optimise_with_ast(self, code, input_data, expected_output):
        """
        Optimises the given code using Abstract Syntax Tree (AST) transformations and rule-based optimisations.

        Args:
            code (str): The source code to be optimised.
            input_data (Any): The input data to be used for validating the optimised code.
            expected_output (Any): The expected output to be used for validating the optimised code.

        Returns:
            Tuple[Optional[str], Dict[str, Any]]: A tuple containing the optimised code (or None if the optimisation is invalid)
            and a dictionary of benchmarks comparing the original and optimised code.
        """
        self.ast_parser.parse_ast(code)

        # Apply rule based optimisations
        self.rule_based_optimisations.remove_unused_variables()
        self.rule_based_optimisations.constant_folding()
        self.rule_based_optimisations.dead_code_elimination()

        optimised_code = self.ast_parser.get_source()

        # Benchmark
        benchmarks = self.benchmark_code(code, optimised_code)

        # Validate
        is_valid = self.validate_output(
            optimised_code, input_data, expected_output)

        if is_valid:
            return optimised_code, benchmarks
        else:
            return None, benchmarks
