import ast
import astpretty

class ASTParser:
    """
    A class used to parse the Abstract Syntax Tree (AST) from source code.
    
    The class uses various methods to parse source code into an AST,
    visualising the AST structure, walking through nodes, applying optimisations
    and converting nodes back into source code.

    Attributes
    ----------
    source_code : str
        The source code to be parsed.
    tree : ast.AST
        The parsed AST of the source code.

    Methods
    -------
    __init__(self, source_code: str):
        Initialises the ASTParser with the given source code and parses the AST.

    parse_ast(source_code: str) -> ast.AST:
        Parses the given source code into an Abstract Syntax Tree (AST).
        Raises a SyntaxError if the source code is invalid.

    visualise_ast(tree: ast.AST) -> None:
        Visualises the given AST using the `astpretty` library.

    walk_and_apply(optimisation_func: function) -> None:
        Walks through the AST and applies the given optimisation function
        to each node of the AST.

    get_node_source(node: ast.AST) -> str:
        Converts an AST node back to the corresponding source code.
    """
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.tree = ast.parse(source_code)

    def parse_ast(self, source_code):
        """
        Parses the given source code into an Abstract Syntax Tree (AST).

        Args:
            source_code (str): The source code to be parsed.

        Returns:
            ast.AST: The parsed AST if the source code is valid.

        Raises:
            SyntaxError: If the source code contains a syntax error.
        """
        self.tree = ast.parse(source_code)
        return self.tree
        
    def visualise_ast(self):
        """
        Visualises the abstract syntax tree (AST) using the astpretty library.

        Args:
            tree (AST): The abstract syntax tree to be visualised.

        Returns:
            None
        """
        astpretty.pprint(self.tree)
        
    def walk_and_apply(self, optimisation_func):
        """
        Walks through the AST and applies the given optimisation to the node
        
        Args:
            optimisation_func (function):
                The optimisation to apply to the AST nodes.
        """
        for node in ast.walk(self.tree):
            optimisation_func(self, node)
            
    def get_node_source(self, node):
            """
            Converts an AST node back to source code.

            Args:
                node (ast.AST):
                    The AST node to convert.

            Returns:
                str:
                    The source code corresponding to the AST node.
            """
            return ast.unparse(node)
        
    def get_source(self):
        """
        Converts the entire AST back to source code.

        Returns:
            str:
                The source code corresponding to the entire AST.
        """
        return ast.unparse(self.tree)
    
if __name__ == "__main__":
    # Display of `visualise_ast` & constructor methods.
    sample_code = "x = 10\nprint(x)"
    parser = ASTParser(sample_code)
    if parser.tree:
        parser.visualise_ast()