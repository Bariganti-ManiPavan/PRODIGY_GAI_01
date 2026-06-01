import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

def main():
    print("\n--- Task 01: GPT-2 Fine-Tuning Started ---")

    # 1. Prepare Custom Dataset
    data = [
        "Artificial intelligence is transforming the world of technology. <|endoftext|>",
        "GPT-2 is a transformer model that excels at generating coherent text. <|endoftext|>",
        "Fine-tuning allows a pre-trained model to adapt to a specific style and tone. <|endoftext|>",
        "Data science and machine learning are highly demanded skills in 2026. <|endoftext|>"
    ]
    dataset = Dataset.from_dict({"text": data})

    # 2. Configure Tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

    # 3. Load Pre-trained GPT-2 Model
    print("Loading pre-trained GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 4. Set Training Arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        num_train_epochs=3, 
        per_device_train_batch_size=1, 
        logging_steps=1,
        save_steps=5,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("Training in progress... Please wait...")
    trainer.train()
    print("--- Training Successfully Completed! ---")

    # 5. Save Fine-Tuned Model and Tokenizer
    model.save_pretrained("./my_finetuned_gpt2")
    tokenizer.save_pretrained("./my_finetuned_gpt2")
    print("Model perfectly saved to './my_finetuned_gpt2' folder.\n")

if __name__ == "__main__":
    main()