import ast
import builtins
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
        
    remove_node(self, node) -> None:
        Removes a specific node from the AST.
        
    TODO: add docs for other methods
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
    
    def remove_node(self, node):
        """
        Removes the specified node from the AST (Abstract Syntax Tree).
        
        Args:
            node (ast.AST): The node to be removed from the AST.
            
        Returns:
            None
        """
        parent = self._find_parent(node)
        if parent:
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    if node in value:
                        value.remove(node)
                        return
                elif value is node:
                    setattr(parent, field, None)
                    return

            
    def find_assignments(self):
        """
        Finds all direct variable assignments (e.g., `=`) in the AST.
        Note: This method does not find augmented assignments (e.g., `+=`).

        Returns:
            dict: A dictionary with variable names as keys and assignment nodes as values.
        """
        assignments = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node
        return assignments
    
    def find_used_variables(self):
        """
        Finds all variables that are used in the AST.

        Returns:
            set: A set of variable names that are used in the AST.
        """
        used_vars = set()
        builtins_set = set(dir(builtins))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in builtins_set:
                    used_vars.add(node.id)
        return used_vars
    
    def fix_locations(self):
        """
        Fixes the locations of the nodes in the abstract syntax tree (AST).

        Returns:
            None
        """
        self.tree = ast.fix_missing_locations(self.tree)
        
    def _find_parent(self, node):
        """
        Find the parent node of a given AST node.

        Args:
            node (ast.AST): The AST node for which to find the parent.

        Returns:
            ast.AST: The parent node of the given AST node, or None if no parent is found.
        """
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                if child is node:
                    return parent
    
    # TODO: Unit testing from here down.
    def create_node(self, node_type, **kwargs):
        """
        Creates a new AST node of the given type with the specified attributes.
        
        Args:
            node_type (str): The type of the AST node to create.
            **kwargs: The attributes to set on the new node.
            
        Returns:
            ast.AST: A new AST node of the given type with the specified attributes.
        """
        node_class = getattr(ast, node_type)
        return node_class(**kwargs)
    
    def find_constant_expressions(self):
        """
        Finds constant expressions in the AST and computes their values.
        
        Returns:
            list: A list of tuples where each tuple contains the original node and the computed value.
        """
        constant_expressions = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                try:
                    new_value = eval(compile(ast.Expression(node), '', 'eval'))
                    constant_expressions.append((node, new_value))
                except Exception:
                    continue

        return constant_expressions

    def replace_node(self, old_node, new_node):
        """
        Replaces an old node with a new node in the AST.
        
        Args:
            old_node (ast.AST): The node to be replaced.
            new_node (ast.AST): The new node to replace the old node.
            
        Returns:
            None
        """
        # Copy line number and column offset from the old node
        # new_node.lineno = old_node.lineno
        # new_node.col_offset = old_node.col_offset
        
        parent = self._find_parent(old_node)
        if parent:
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if item is old_node:
                            value[i] = new_node
                            self.fix_locations()
                            return
                elif value is old_node:
                    setattr(parent, field, new_node)
                    self.fix_locations()
                    return
                
    def find_unreachable_code(self):
        """
        Finds unreachable code in the AST.
        
        Returns:
            unreachable_nodes (list): A list of nodes that are unreachable.
        """
        unreachable_nodes = []
        for node in ast.walk(self.tree):
            
            # Code after return, raise, break, or continue
            if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                parent = self._find_parent(node)
                if parent:
                    if isinstance(parent, ast.FunctionDef): # function
                        unreachable_nodes.extend(parent.body[parent.body.index(node) + 1:])
                    elif isinstance(parent, ast.For) or isinstance(parent, ast.While): # loop
                         unreachable_nodes.extend(parent.body[parent.body.index(node) + 1:])
                    elif isinstance(parent, ast.If): # if statements
                        if node in parent.body:
                            unreachable_nodes.extend(parent.body[parent.body.index(node) + 1:])
                        elif node in parent.orelse:
                             unreachable_nodes.extend(parent.orelse[parent.orelse.index(node) + 1:])
                    
            # Unreachable code in if statements by constant conditions
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                if not node.test.value:
                    unreachable_nodes.extend(node.body)
                else:
                    unreachable_nodes.extend(node.orelse)
                    
                    
            # Unreachable code in Functions
            if isinstance(node, ast.FunctionDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Constant) and not stmt.test.value:
                        unreachable_nodes.extend(stmt.body)
                    elif isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Constant) and stmt.test.value:
                        unreachable_nodes.extend(stmt.orelse)
                        
            # Unreachable code in loops
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and not node.test.value:
                    unreachable_nodes.extend(node.body)
        
            if isinstance(node, ast.For):
                # If the range in the for loop is empty, the body is unreachable
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range' and isinstance(node.iter.args[0], ast.Constant) and node.iter.args[0].value == 0:
                        unreachable_nodes.extend(node.body)
                
        return unreachable_nodes
                
        

    
if __name__ == "__main__":
    sample_code = """
x = 5 + 3
y = 10 * 2
z = y - 5
w = x + z

if x == 7:
    c = 42
else:
    d = 84

if False:
    a = 42

if True:
    b = 42

while False:
    e = 99

for i in range(0):
    f = 100

def test_func():
    if False:
        g = 88
    return 10
    gg = 42

print(x, y, z, w)
""" # Sample code for demo

    parser = ASTParser(sample_code) # Constructor

    print("Original AST:")
    parser.visualise_ast()  # Display of visualise_ast
    
    constant_expressions = parser.find_constant_expressions() # Find constant expressions
    print("\nDetected Constant Expressions:")
    for node, value in constant_expressions:
        print(f"Expression: {parser.get_node_source(node)} -> Computed Value: {value}")
    
    # node replacement process - CONSTANT FOLDING - in `rule_based_optimisations.py`
    for old_node, new_value in constant_expressions:
        new_constant_node = parser.create_node('Constant', value=new_value)
        parser.replace_node(old_node, new_constant_node)
        
    # display removed unreachable code
    unreachable_nodes = parser.find_unreachable_code()
    for node in unreachable_nodes:
            parser.remove_node(node)
    
    print("\nOptimised AST:")
    parser.visualise_ast()  # Display AST after modification

    print("\nOptimised Code:")
    print(parser.get_source())  # Show optimised source code