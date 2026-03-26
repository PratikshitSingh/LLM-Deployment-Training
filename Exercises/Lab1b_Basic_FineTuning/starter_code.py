"""
Lab 1b: Basic Fine-Tuning for Sentiment Analysis with DistilBERT

This script demonstrates a basic fine-tuning workflow for sentiment analysis using a local copy
of "distilbert-base-uncased-finetuned-sst-2-english". [EXERCISE]
Below, fill in the missing training implementation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], 
                     truncation=True, 
                     padding="max_length",
                     max_length=128,
                     )

def main():
    model_name = "../local_models/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Create a dummy dataset
    data = {
        "text": [
            "This is a very bad product. I am extremely disappointed.",
            "I absolutely love this! It works great and I am very happy."
        ],
        "label": [0, 1]
    }
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none"  # No API reporting
        # [EXERCISE] Add any additional training parameters you think are necessary.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        # [EXERCISE] Optionally, add evaluation metrics or a compute_metrics function.
    )
    
    print("Starting fine-tuning for sentiment analysis...")
    # please fill here with code to call trainer.train()
    trainer.train()
    print("Fine-tuning complete!")

    # make a prediction on a new sentence
    new_sentence = "This is shit!"
    new_sentence_tokenized = tokenizer(new_sentence, return_tensors="pt")
    
    model.eval()
    model_device = next(model.parameters()).device
    new_sentence_tokenized = new_sentence_tokenized.to(model_device)
    with torch.no_grad():
        prediction = model(**new_sentence_tokenized)
    predicted_class = torch.argmax(prediction.logits, dim=-1).item()
    labels = ["NEGATIVE", "POSITIVE"]
    print(f"Prediction for '{new_sentence}': {labels[predicted_class]} (logits: {prediction.logits}), (new_sentence_tokenized: {new_sentence_tokenized})")

if __name__ == "__main__":
    main()
