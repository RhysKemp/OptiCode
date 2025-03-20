import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from ai.dataset import CustomDataset
from ai.model import Seq2SeqModel
from engine.ast_parser import ASTParser
from tqdm import tqdm

# TODO: need to fix model training


def train_model(dataset_path, epochs=10, batch_size=32, learning_rate=0.001):
    # Load
    dataset = CustomDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define
    input_dim = output_dim = len(dataset[0][1])
    hidden_dim = 128
    model = Seq2SeqModel(input_dim, hidden_dim, output_dim)
    criterion = MSELoss()
    optimiser = Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for i, (inefficient_ast, efficient_ast) in enumerate(dataloader):

                # convert to tensors
                inefficient_ast = torch.tensor(
                    inefficient_ast, dtype=torch.float32
                ).unsqueeze(1)
                efficient_ast = torch.tensor(
                    efficient_ast, dtype=torch.float32
                ).unsqueeze(1)

                optimiser.zero_grad()

                # forward pass
                outputs = model(inefficient_ast, efficient_ast)
                loss = criterion(outputs, efficient_ast)

                # Backwards pass and optimise
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print every 10 batches
                    print(
                        f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}"
                    )
                    running_loss = 0.0

        print("Finished Training")
        # Save model
        torch.save(model.state_dict(), "ai/models/model.pth")


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC/TEST_DEV")
    train_model(dataset_path)
