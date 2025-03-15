import ast
import astpretty

def parse_ast(source_code):
    """
    Parses the given source code into an Abstract Syntax Tree (AST).

    Args:
        source_code (str): The source code to be parsed.

    Returns:
        ast.AST: The parsed AST if the source code is valid.

    Raises:
        SyntaxError: If the source code contains a syntax error.
    """
    tree = ast.parse(source_code)
    return tree
    
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
    # Display of function
    sample_code = "x = 10\nprint(x)"
    tree = parse_ast(sample_code)
    if tree:
        visualise_ast(tree)