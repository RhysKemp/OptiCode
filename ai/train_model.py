import os
import time
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from ai.dataset import CodeDataset
from sacrebleu.metrics import BLEU


def compute_metrics(eval_preds, tokeniser):
    """
    Compute BLEU score for the model's predictions.

    Args:
        eval_preds (tuple): A tuple containing predicted logits and target labels.

    Returns:
        dict: A dictionary with 'bleu' score.
    """
    logits, labels = eval_preds
    logits = logits[0]

    # Decode predictions and labels
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded_preds = tokeniser.batch_decode(
        predicted_ids, skip_special_tokens=True)
    decoded_labels = tokeniser.batch_decode(labels, skip_special_tokens=True)

    bleu = BLEU()
    bleu_score = bleu.corpus_score(decoded_preds, decoded_labels).score

    return {'bleu': bleu_score}


def load_datasets(train_dir: str, test_dir: str, tokeniser, max_length: int):
    """
    Load training and test datasets.

    Args:
        train_dir (str): The directory containing the training dataset.
        test_dir (str): The directory containing the test dataset.
        tokeniser: The tokeniser to be used for processing the datasets.
        max_length (int): The maximum length of the tokenised sequences.

    Returns:
        tuple: A tuple containing the training dataset and the test dataset.
    """

    print(f"Loading training dataset from {train_dir}")
    training_data = CodeDataset(
        dataset_dir=train_dir, tokeniser=tokeniser, max_length=max_length)
    print(f"Loaded {len(training_data)} training examples")

    print(f"Loading test dataset from {test_dir}")
    test_data = CodeDataset(dataset_dir=test_dir,
                            tokeniser=tokeniser, max_length=max_length)
    print(f"Loaded {len(test_data)} test examples")

    return training_data, test_data


def data_collator(features, tokeniser, max_length, device):
    """
    Collates and pads a batch of features for model training.

    Unused method, `DataCollatorForSeq2Seq` used instead.

    Args:
        features (list of dict): A list of dictionaries containing the features for each sample.
            Each dictionary should have the keys "input_ids", "attention_mask", and "labels".
        tokeniser (PreTrainedTokenizer): The tokeniser used to process the input data.
            This is used to get the padding token ID.
        max_length (int): The maximum sequence length to which the input sequences will be truncated.

    Returns:
        dict: A dictionary containing the padded and truncated tensors for "input_ids", "attention_mask", and "labels".
            - "input_ids" (torch.Tensor): Padded and truncated input IDs.
            - "attention_mask" (torch.Tensor): Padded and truncated attention masks.
            - "labels" (torch.Tensor): Padded and truncated labels.
    """

    # Extract raw features
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]

    # Pad sequences correctly
    input_ids = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                             for seq in input_ids], batch_first=True, padding_value=tokeniser.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                                  for seq in attention_mask], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(seq, dtype=torch.long) if not isinstance(seq, torch.Tensor) else seq
                          for seq in labels], batch_first=True, padding_value=-100)

    # Pin tensors before moving to GPU
    input_ids = input_ids.pin_memory()
    attention_mask = attention_mask.pin_memory()
    labels = labels.pin_memory()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train_model(dataset_dir: str, output_dir: str, model_name: str = "salesforce/codet5-base", max_length: int = 512, epochs: int = 3, batch_size: int = 8):
    """
    Trains a model on a given dataset and saves the trained model and tokeniser.

    Args:
        dataset_dir (str): Directory containing the dataset.
        output_dir (str): Directory to save the trained model and tokeniser.
        model_name (str, optional): Name of the pre-trained T5 model to use. Defaults to "salesforce/codet5-small".
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        epochs (int, optional): Number of training epochs. Defaults to 3.
        batch_size (int, optional): Batch size for training and evaluation. Defaults to 8.

    Returns:
        tuple: A tuple containing the trained model and tokeniser.
    """

    print(f"Starting model training process with {model_name}")

    print(f"Loading tokeniser and model...")
    start_time = time.time()
    tokeniser = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(
        f"Loaded tokeniser and model in {time.time() - start_time:.2f} seconds")

    # Load dataset
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    print(f"Loading and processing dataset from {dataset_dir}")
    train_dataset, val_dataset = load_datasets(
        train_dir=train_dir, test_dir=test_dir, tokeniser=tokeniser, max_length=max_length)

    # display train/val size
    print(f"Splitting dataset into training and validation sets")
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(
        f"Dataset loaded: {train_size} training examples, {val_size} validation examples in {time.time() - start_time:.2f} seconds")

    # Define training args
    print(f"Configuring training parameters")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        report_to=["tensorboard"],
        overwrite_output_dir=True,
        disable_tqdm=False,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Display training params
    print(f"Training model for {epochs} epochs")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Evaluation every {training_args.eval_steps} steps")
    print(f"   - Saving model every {training_args.save_steps} steps")

    data_collator_fn = DataCollatorForSeq2Seq(tokeniser, model=model)

    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokeniser,
        compute_metrics=compute_metrics
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    minutes, seconds = divmod(training_time, 60)
    print(
        f"Training completed in {int(minutes)} minutes {seconds:.2f} seconds")

    # Save model
    print(f"Saving final model to {os.path.join(output_dir, 'final_model')}")
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokeniser.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"Model saved successfully")

    return model, tokeniser


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(
        __file__), "ai/datasets/GEC")
    output_dir = os.path.join(os.path.dirname(
        __file__), "ai/models")

    trained_model, trained_tokeniser = train_model(
        dataset_dir=dataset_dir, output_dir=output_dir, epochs=3, batch_size=2, max_length=768)
