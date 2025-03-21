import torch
import os
from utils.file_loader import load_text_file

from transformers import T5ForConditionalGeneration, RobertaTokenizer

model_dir = os.path.join(os.path.dirname(
    __file__), "ai/models/final_model")


tokeniser = RobertaTokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

test_code_dir = os.path.join(os.path.dirname(
    __file__), "ai/datasets/GEC/TEST_DEV/0006/Acc_tle_solutions/0,374 ms,1480 KB.json")
print(test_code_dir)
test_code = load_text_file(test_code_dir)

print("\n\nOriginal input: ")
print(test_code)

tokenised_code = tokeniser.encode(test_code, return_tensors="pt")
print("\n\ntokenised input: ")
print(tokenised_code)

with torch.no_grad():
    output = model.generate(tokenised_code, max_length=2048)

optimised_code = tokeniser.decode(output[0], skip_special_tokens=True)

print("Optimised Code:", optimised_code)
