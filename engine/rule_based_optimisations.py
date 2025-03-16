import ast_parser

"""
This class contains the manual, rule-based optimisation implementations used in the OptiCode project.

Overview:
The optimisations in this class are based on a set of predefined rules designed to improve the performance of code manually. 

These optimisations target execution speed, memory usage, and energy consumption.

Future Work:
In future iterations, the goal is to integrate AI-driven optimisations based on machine learning models that should replace the current iteration and this folder.

This file will remain here just as a display for the initial pipeline and as a backup if an AI model fails.
"""

class RuleBasedOptimisations():
    """
    This class contains the manual, rule-based optimisation implementations used in the OptiCode project.
    
    Overview:
    The optimisations in this class are based on a set of predefined rules designed to improve the performance of code manually. These optimisations target execution speed, memory usage, and energy consumption.
    
    This class focuses on transformations such as:
        - Removing redundant variable assignments
        - 
        -
    
    """
    
    def __init__(self, ast_parser):
        self.ast_parser = ast_parser
        
    def remove_unused_variables(self):
        """
        Removes unused variable assignments from the abstract syntax tree (AST).
        
        This method identifies all variable assignments and used variables within the AST.
        It then removes any assignments that are not used elsewhere in the code.
        
        Prints:
            A message indicating the node that was removed.
            
        Returns:
            None
        """
        assignments = self.ast_parser.find_assignments()
        used_vars = self.ast_parser.find_used_variables()
        
        for var, node in assignments.items():
            if var not in used_vars:
                self.ast_parser.remove_node(node)
                print(f"Removed node: {node.value.n}")
        
        self.ast_parser.fix_locations()
        
    def loop_unrolling(self):
        pass
        
        
if __name__ == "__main__":
    
    # Basic test of removing unused variable
    sample_code = """
x = 10
y = 20
print(x)
"""
    parser = ast_parser.ASTParser(sample_code)
    optimiser = RuleBasedOptimisations(parser)
    print(sample_code)
    optimiser.remove_unused_variables()
    print(parser.get_source())