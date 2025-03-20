import unittest
from engine.rule_based_optimisations import RuleBasedOptimisations
from engine.ast_parser import ASTParser


class TestRuleBasedOptimisations(unittest.TestCase):

    def setUp(self):
        self.sample_code = """
x = 10
y = 20
print(x)
"""
        self.parser = ASTParser()
        self.optimiser = RuleBasedOptimisations(self.parser)
        self.parser.parse_ast(self.sample_code)

    def test_remove_unused_variables(self):
        self.optimiser.remove_unused_variables()
        expected_code = """
x = 10
print(x)
"""
        self.assertEqual(self.parser.get_source().strip(), expected_code.strip())

    def test_constant_folding(self):
        sample_code = """
x = 10 + 20
y = x * 2
print(y)
"""
        self.parser.parse_ast(sample_code)
        optimiser = RuleBasedOptimisations(self.parser)
        optimiser.constant_folding()
        expected_code = """
x = 30
y = x * 2
print(y)
"""
        self.assertEqual(self.parser.get_source().strip(), expected_code.strip())

    def test_full_class(self):
        """
        Test the full class of rule-based optimisations on a sample code.

        Asserts:
            The optimised code matches the expected code.
        """
        sample_code = """
x = 10 + 20
y = x * 2
if False:
    b = 42
z = 3
print(y)
"""
        self.parser.parse_ast(sample_code)
        optimiser = RuleBasedOptimisations(self.parser)
        for method_name in dir(optimiser):
            if callable(getattr(optimiser, method_name)) and not method_name.startswith(
                "__"
            ):
                getattr(optimiser, method_name)()
                print(f"Running: {method_name}")
        expected_code = """
x = 30
y = x * 2
if False:
print(y)
"""
        print(self.parser.get_source().strip())
        self.assertEqual(self.parser.get_source().strip(), expected_code.strip())


if __name__ == "__main__":
    unittest.main()
