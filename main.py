import os
from ai.train_model import train_model
from engine.optimiser import CodeOptimiser
from utils.file_loader import load_input_output_data, load_text_file

if __name__ == "__main__":

    # Train model call
    # dataset_dir = os.path.join(os.path.dirname(
    #     __file__), "ai/datasets/GEC")
    # output_dir = os.path.join(os.path.dirname(
    #     __file__), "ai/models")

    # trained_model, trained_tokeniser = train_model(
    #     dataset_dir=dataset_dir, output_dir=output_dir, epochs=3, batch_size=4, max_length=512)

    # Pipeline display, future work would implement UI
    model_dir = os.path.join(os.path.dirname(
        __file__), "ai/models/final_model")
    optimiser = CodeOptimiser(model_dir)

    test_code_dir = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC/TEST_DEV/0000")
    test_code_file = test_code_dir + "/Acc_tle_solutions/0,186 ms,8 KB.json"
    test_code = load_text_file(test_code_file)

    input_output_dir = test_code_dir + "/input_output.json"
    input_data, output_data = load_input_output_data(input_output_dir)

    input_str = input_data[0]
    output_str = output_data[0]

    optimised_code, benchmarks = optimiser.optimise_code(
        test_code, input_str, output_str)

    if optimised_code:
        print("Optimised Code:", optimised_code)
        print("Benchmarks:", benchmarks)
    else:
        print(
            "Optimisation failed. The generated code did not produce the expected output.")
