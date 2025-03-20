import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from engine.ast_parser import ASTParser


def test_model(dataset_dir, model_dir):
    # Load the saved model and tokenizer
    print(f"Loading model and tokenizer from {model_dir}")
    tokeniser = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    # Initialize AST parser
    parser = ASTParser()

    # Test the model on a sample input
    test_code = "x = 10\ny = 20\nz = 30\nprint(x)\nprint(z)"
    print("Original Code:")
    print(test_code)

    # Parse the code to AST and linearise it
    parser.parse_ast(test_code)
    linearised_ast = parser.linearise_ast()

    print("\nLinearised AST Input:")
    print(linearised_ast)

    # Convert to model input format using linearised AST
    inputs = tokeniser(linearised_ast, return_tensors="pt",
                       padding=True, truncation=True)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    # Decode the output
    generated_code = tokeniser.decode(output_ids[0], skip_special_tokens=True)

    # Parse and convert back to code if necessary
    # If your model outputs linearised AST, you might need to convert it back to code

    print("\nGenerated Code:")
    print(generated_code)


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC/TEST_DEV")
    model_dir = os.path.join(os.path.dirname(
        __file__), "ai/models/final_model")
    test_model(dataset_dir, model_dir)
