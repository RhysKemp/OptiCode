import tokenize
import unittest
from engine.tokenization import tokenize_code

class TestTokenization(unittest.TestCase):
    
    def setUp(self):
        """Setup test cases for tokenization"""
        self.test_cases = [
            {
                "name": "simple_code",
                "input": "x = 10\nprint(x)",
                "expected": [
                    (tokenize.NAME, 'x'),
                    (tokenize.OP, '='),
                    (tokenize.NUMBER, '10'),
                    (tokenize.NEWLINE, '\n'),
                    (tokenize.NAME, 'print'),
                    (tokenize.OP, '('),
                    (tokenize.NAME, 'x'),
                    (tokenize.OP, ')'),
                ]
            },
            {
                "name": "empty_code",
                "input": "",
                "expected": []
            },
            {
                "name": "invalid_code",
                "input": "x = 10 + @5",
                "expected": [
                    (tokenize.NAME, 'x'),
                    (tokenize.OP, '='),
                    (tokenize.NUMBER, '10'),
                    (tokenize.OP, '+'),
                    (tokenize.OP, '@'),
                    (tokenize.NUMBER, '5')
                    
                ]
            }
        ]
        
    def test_tokenize_valid_code(self):
        """Test tokenizing valid code"""
        for case in self.test_cases:
            with self.subTest(case=case["name"]):
                result = tokenize_code(case["input"])
                token_values = [(token[0], token[1]) for token in result if token[1] != ''] # Filter empty string tokens
                expected_values = case["expected"]
                
                # Assert
                self.assertEqual(len(token_values), len(expected_values), # Check len of list
                                 f"Mismatch in number of tokens for {case['name']}. Expected: {len(expected_values)}, Got: {len(token_values)}")
                
                for i, (actual_token, expected_token) in enumerate(zip(token_values, expected_values)): # Check content of tokens
                    self.assertEqual(actual_token[0], expected_token[0], 
                                     f"Token type mismatch at index {i} in {case['name']}. Expected: {expected_token[0]}, Got: {actual_token[0]}")
                    self.assertEqual(actual_token[1], expected_token[1], 
                                     f"Token value mismatch at index {i} in {case['name']}. Expected: '{expected_token[1]}', Got: '{actual_token[1]}'")
                
                if len(token_values) != len(expected_values): # catch-all
                    print(f"Expected tokens: {expected_values}")
                    print(f"Actual tokens:   {token_values}")
                
                
if __name__ == "__main__":
    unittest.main()