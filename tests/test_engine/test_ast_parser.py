import unittest
from engine.ast_parser import parse_ast, visualise_ast
import ast

class TestASTParser(unittest.TestCase):

    def test_parse_ast_valid_code(self):
        sample_code = "x = 10\nprint(x)"
        tree = parse_ast(sample_code)
        self.assertIsInstance(tree, ast.AST)

    def test_parse_ast_invalid_code(self):
        sample_code = "x = 10\nprint(x"
        tree = parse_ast(sample_code)
        self.assertIsNone(tree)

    # def test_visualise_ast(self):
    #     # Optional manual test
    #     sample_code = "x = 10\nprint(x)"
    #     tree = parse_ast(sample_code)
    #     self.assertIsNotNone(tree)
    #     try:
    #         visualise_ast(tree)
    #     except Exception as e:
    #         self.fail(f"visualise_ast raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()