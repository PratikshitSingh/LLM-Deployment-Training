"""
Lab 6: Model Quantization

This script demonstrates three quantization approaches for t5-small:
1. Training-Aware Quantization (simulation)
2. bfloat16 Quantization
3. int8 Dynamic Quantization

Usage: python starter_code.py [taq|bfloat16|int8]
"""
import gc
import os
import sys
import torch
import psutil
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def count_all_params(model):
    """Count parameters including those inside quantized packed modules."""
    total_params = 0
    total_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.Linear):
            w = module.weight()
            total_params += w.numel()
            total_bytes += w.nelement() * w.element_size()
            if module.bias() is not None:
                b = module.bias()
                total_params += b.numel()
                total_bytes += b.nelement() * b.element_size()

    for p in model.parameters():
        total_params += p.numel()
        total_bytes += p.nelement() * p.element_size()

    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_bytes += buffer_bytes

    return total_params, total_bytes


def get_param_size_mb(model):
    """Approach 2: count params and do the math."""
    _, total_bytes = count_all_params(model)
    return total_bytes / (1024 * 1024)


def get_process_ram_mb():
    """Approach 3: process.memory_info() RSS."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def profile_forward(model, input_ids):
    """Approach 1: torch.profiler to profile a forward pass on CPU."""
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=20)
    return prof


def benchmark(label, model, input_ids):
    num_params, total_bytes = count_all_params(model)
    param_size = total_bytes / (1024 * 1024)
    ram = get_process_ram_mb()
    prof = profile_forward(model, input_ids)
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  Param size:   {param_size:.2f} MB")
    print(f"  Num params:   {num_params:,}")
    print(f"  Process RSS:  {ram:.2f} MB")
    print(f"\n  Profiler (top ops by CPU memory):")
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage", row_limit=10
    ))
    return param_size


def quantize_model(model, mode):
    if mode == "bfloat16":
        model = model.to(torch.bfloat16)
        print("Model converted to bfloat16.")
    elif mode == "int8":
        torch.backends.quantized.engine = "qnnpack"
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Model dynamically quantized to int8.")
    elif mode == "taq":
        print("Training-Aware Quantization selected (simulation).")
    else:
        print("Invalid mode; returning original model.")
    return model

def load_or_quantize(model, mode, model_name):
    output_dir = f"./quantized_model_{mode}"

    if mode == "int8":
        save_path = os.path.join(output_dir, "model_int8.pt")
        if os.path.exists(save_path):
            print(f"Loading saved int8 model from {save_path}")
            torch.backends.quantized.engine = "qnnpack"
            quantized = torch.ao.quantization.quantize_dynamic(
                AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True),
                {torch.nn.Linear}, dtype=torch.qint8
            )
            quantized.load_state_dict(torch.load(save_path, weights_only=False))
            return quantized

    elif mode in ("bfloat16", "taq"):
        if os.path.isdir(output_dir) and os.path.exists(
            os.path.join(output_dir, "model.safetensors")
        ):
            print(f"Loading saved model from {output_dir}")
            return AutoModelForSeq2SeqLM.from_pretrained(
                output_dir, local_files_only=True
            )

    print("No saved quantized model found, quantizing on the fly...")
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    quantized = quantize_model(model, mode)
    if mode == "int8":
        os.makedirs(output_dir, exist_ok=True)
        torch.save(quantized.state_dict(),
                    os.path.join(output_dir, "model_int8.pt"))
    else:
        quantized.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")
    return quantized


def main():
    if len(sys.argv) < 2:
        print("Usage: python starter_code.py [taq|bfloat16|int8]")
        sys.exit(1)
    mode = sys.argv[1]
    model_name = "../local_models/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

    sample_input = tokenizer("Summarize: Hello world", return_tensors="pt").input_ids

    original_params = benchmark("ORIGINAL MODEL", model, sample_input)

    del model
    gc.collect()

    print(f"\nApplying quantization mode: {mode}")
    quantized_model = load_or_quantize(None, mode, model_name)

    quantized_params = benchmark(f"QUANTIZED MODEL ({mode})", quantized_model, sample_input)

    delta = quantized_params - original_params
    if original_params > 0:
        pct = abs(delta) / original_params * 100
        direction = "reduction" if delta <= 0 else "increase"
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  Param size: {original_params:.2f} MB -> {quantized_params:.2f} MB "
              f"({delta:+.2f} MB, {pct:.1f}% {direction})")

if __name__ == "__main__":
    main()
