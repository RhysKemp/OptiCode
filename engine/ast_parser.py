import ast
import astpretty

def parse_ast(source_code):
    """
    Parses the given source code into an Abstract Syntax Tree (AST).

    Args:
        source_code (str): The source code to be parsed.

    Returns:
    
        ast.AST: The parsed AST if the source code is valid.
        
        None: If there is a syntax error in the source code.

    Raises:
        SyntaxError: If the source code contains a syntax error.
    """
    try:
        tree = ast.parse(source_code)
        return tree
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return None
    
def visualise_ast(tree):
    """
    Visualises the abstract syntax tree (AST) using the astpretty library.

    Args:
        tree (AST): The abstract syntax tree to be visualised.

    Returns:
        None
    """
    astpretty.pprint(tree)
    
if __name__ == "__main__":
    sample_code = "x = 10\nprint(x)"
    tree = parse_ast(sample_code)
    if tree:
        visualise_ast(tree)