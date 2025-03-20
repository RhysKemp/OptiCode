import os
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from ai.dataset import LinearDataset
from tqdm import tqdm


def data_collator(features, tokeniser, max_length):
    # Extract raw features
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]

    # Optional debugging - only if needed
    # for i, f in enumerate(features):
    #     print(f"Example {i} input_ids length: {len(f['input_ids'])}")
    #     print(f"Example {i} attention_mask length: {len(f['attention_mask'])}")
    #     print(f"Example {i} labels length: {len(f['labels'])}")

    # Pad sequences correctly
    input_ids = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                             for seq in input_ids], batch_first=True, padding_value=tokeniser.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                                  for seq in attention_mask], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                          for seq in labels], batch_first=True, padding_value=-100)

    # Ensure the tensors are correctly shaped (batch_size, max_length)
    if input_ids.dim() > 2:
        input_ids = input_ids[:, :, 0]  # Remove extra dimension
    if attention_mask.dim() > 2:
        attention_mask = attention_mask[:, :, 0]  # Remove extra dimension
    if labels.dim() > 2:
        labels = labels[:, :, 0]  # Remove extra dimension

    # Now truncate to max_length
    input_ids = input_ids[:, :max_length]
    attention_mask = attention_mask[:, :max_length]
    labels = labels[:, :max_length]

    # Print final shapes after corrections
    print(f"Final input_ids shape: {input_ids.shape}")
    print(f"Final attention_mask shape: {attention_mask.shape}")
    print(f"Final labels shape: {labels.shape}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_model(dataset_dir: str, output_dir: str, model_name: str = "t5-small", max_length: int = 512, epochs: int = 5, batch_size: int = 8):
    print(f"Starting model training process with {model_name}")

    print(f"Loading tokeniser and model...")
    start_time = time.time()
    tokeniser = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print(
        f"Loaded tokeniser and model in {time.time() - start_time:.2f} seconds")

    # Load dataset
    print(f"Loading and processing dataset from {dataset_dir}")
    start_time = time.time()
    dataset = LinearDataset(dataset_dir=dataset_dir,
                            tokeniser=tokeniser, max_length=max_length)
    print(
        f"Dataset loaded with {len(dataset)} examples in {time.time() - start_time:.2f} seconds")

    # Split validation set
    print(f"Splitting dataset into training and validation sets")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(
        f"Dataset split: {train_size} training examples, {val_size} validation examples")

    # Define training args with better progress reporting
    print(f"Configuring training parameters")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        logging_first_step=True,
        report_to=["tensorboard"],
        disable_tqdm=False,
    )

    # Train with clear indication of progress
    print(f"Training model for {epochs} epochs")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Evaluation every {training_args.eval_steps} steps")
    print(f"   - Saving model every {training_args.save_steps} steps")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda features: data_collator(
            features, tokeniser, max_length),
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    start_time = time.time()
    train_output = trainer.train()
    print(type(train_output), train_output)
    training_time = time.time() - start_time

    minutes, seconds = divmod(training_time, 60)
    print(
        f"Training completed in {int(minutes)} minutes {seconds:.2f} seconds")

    # print(f"Final training loss: {train_output.training_loss:.4f}")

    # Save pre-trained model
    print(f"Saving final model to {os.path.join(output_dir, 'final_model')}")
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokeniser.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"Model saved successfully")

    return model, tokeniser


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC/TEST_DEV")
    output_dir = os.path.join(os.path.dirname(
        __file__), "ai/models")

    trained_model, trained_tokeniser = train_model(
        dataset_dir=dataset_dir, output_dir=output_dir)
