"""
Lab 4: Efficient Distillation using a Custom Training Loop

This script distills a teacher model (flan-t5-xl) into a student model (t5-small) using a custom training loop.
- Teacher is loaded from local files.
- Student is loaded normally.
- The tokenize function sets both "labels" and "decoder_input_ids".
[EXERCISE] Please implement the training loop to update the student model.
"""
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, Dataset
from torch.optim import AdamW
import torch.nn.functional as F

def tokenize_function(example, tokenizer, max_input_length=256, max_target_length=128):
    model_inputs = tokenizer(example["dialogue"], truncation=True, padding="max_length", max_length=max_input_length)
    labels = tokenizer(text_target=example["summary"], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    model_inputs["decoder_input_ids"] = torch.tensor(labels["input_ids"])
    return model_inputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    teacher_model_name = "../local_models/flan-t5-xl"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, local_files_only=True)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name, local_files_only=True).to(device)
    teacher_model.eval()

    student_model_name = "../local_models/t5-small"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, local_files_only=True)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name, local_files_only=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    student_model = get_peft_model(student_model, lora_config).to(device)
    student_model.train()
    
    dataset_dir = "../local_datasets/dialogsum"
    try:
        dataset = load_from_disk(dataset_dir)
        train_dataset = dataset.shuffle(seed=42).select(range(100))
    except Exception as e:
        print(f"Failed to load DialogSum dataset from disk: {e}")

        train_csv = f"{dataset_dir}/train.csv"
        print(f"Loading DialogSum from CSV: {train_csv}")
        dataset = Dataset.from_csv(train_csv)
        num_samples = min(100, len(dataset))
        train_dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, teacher_tokenizer), batched=True)
    
    optimizer = AdamW(student_model.parameters(), lr=3e-5)
    
    num_epochs = 1
    batch_size = 16
    num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"Student model trainable parameters: {num_params:,}")
    print("Starting distillation training loop...")
    
    for epoch in range(num_epochs):
        for i in range(0, len(tokenized_dataset), batch_size):
            batch = tokenized_dataset[i:i + batch_size]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                teacher_logits = teacher_outputs.logits
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 8 == 0:
                print(f"Epoch {epoch+1}, Batch {i // batch_size + 1}, Loss: {loss.item():.4f}")
    
    print("Distillation training complete!")
    
if __name__ == "__main__":
    main()
