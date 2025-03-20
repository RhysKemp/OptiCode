import os
import torch
from ai.dataset import CustomDataset
from ai.model import Seq2SeqModel
from engine.ast_parser import ASTParser


def test_model(dataset_path, model_path):
    # Load dataset
    dataset = CustomDataset(dataset_path)

    # Define model
    # Assuming input and output dimensions are the same
    input_dim = output_dim = len(dataset[0][0])
    hidden_dim = 128
    model = Seq2SeqModel(input_dim, hidden_dim, output_dim)

    # Load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize AST parser
    parser = ASTParser()

    # Test the model on a sample input
    sample_index = 0  # Change this index to test different samples
    inefficient_ast, efficient_ast = dataset[sample_index]

    # Convert AST to tensor
    inefficient_ast_tensor = torch.tensor(
        inefficient_ast, dtype=torch.float32).unsqueeze(0)

    # Generate output
    with torch.no_grad():
        output = model(inefficient_ast_tensor, inefficient_ast_tensor)

    # Convert output tensor to AST
    output_ast = output.squeeze(0).numpy()

    # Convert AST back to source code
    parser.tree = output_ast
    generated_code = parser.get_source()

    # Print the results
    parser.tree = inefficient_ast
    inefficient_code = parser.get_source()

    parser.tree = efficient_ast
    efficient_code = parser.get_source()

    print("Inefficient Code:")
    print(inefficient_code)
    print("\nGenerated Code:")
    print(generated_code)
    print("\nEfficient Code:")
    print(efficient_code)


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC/train")
    model_path = os.path.join(os.path.dirname(
        __file__), "model.pth")
    test_model(dataset_path, model_path)
