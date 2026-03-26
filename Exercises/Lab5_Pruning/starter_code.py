"""
Lab 5: Model Pruning

This script demonstrates how to prune a model (t5-small) to reduce its effective complexity.
We use PyTorch's pruning utilities to remove 20% of weights in Linear layers.
"""
import torch    
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def global_sparsity(model):
    """Check global sparsity: percentage of weights that are exactly zero."""
    zero_weights = sum((p == 0).sum().item() for p in model.parameters())
    total_weights = sum(p.numel() for p in model.parameters())
    sparsity = zero_weights / total_weights * 100
    print(f"Sparsity: {sparsity:.2f}% of weights are zero ({zero_weights}/{total_weights})")
    return sparsity

def per_layer_sparsity(model):
    """Check per-layer sparsity: percentage of weights that are exactly zero."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            zero_weights = sum((p == 0).sum().item() for p in module.weight)
            total_weights = module.weight.numel()
            sparsity = zero_weights / total_weights * 100
            print(f"Layer {name} sparsity: {sparsity:.2f}% of weights are zero ({zero_weights}/{total_weights})")

def inference_sanity_check(model):
    """Check if the model can generate a summary."""
    text = "Hello, how are you?"
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    print(outputs)
    return outputs

def verify_pruning(model):
    sparsity = global_sparsity(model)
    # per_layer_sparsity(model)
    outputs = inference_sanity_check(model)

    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Inference sanity check: {outputs}")

# def model_size_mb(model):
#     param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
#     buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
#     return (param_bytes + buffer_bytes) / 1024**2
#     print("model params+buffers MB:", model_size_mb(model))

def model_size_mb(model):
    param_bytes = sum(
        (p != 0).sum().item() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        (b != 0).sum().item() * b.element_size() for b in model.buffers()
    )
    return (param_bytes + buffer_bytes) / 1024**2


def main():
    model_name = "../local_models/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    print("Before pruning:")
    verify_pruning(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Non-zero model size: {model_size_mb(model):.2f} MB")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.2)
            prune.remove(module, "weight")
    print("=" * 60)
    print("After pruning:")
    verify_pruning(model)
    print(f"Non-zero model size: {model_size_mb(model):.2f} MB")
    model.save_pretrained("./pruned_model")
    print("Pruned model saved to ./pruned_model")

if __name__ == "__main__":
    main()
