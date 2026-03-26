"""
Lab 7: Performance Benchmarking

Benchmarks t5-small with real metrics:
  - Inference latency (warmup + percentiles)
  - BLEU and ROUGE on DialogSum test samples
  - Memory / model-size measurements via psutil and torch
"""
import csv
import io
import os
import statistics
import time

import evaluate
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NUM_EVAL_SAMPLES = 50
NUM_METRICS_SAMPLES_DEFAULT = 20
NUM_WARMUP = 3
NUM_MEASURED = 10

_bleu_metric = None
_rouge_metric = None


def _get_bleu_metric():
    global _bleu_metric
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _get_rouge_metric():
    global _rouge_metric
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _rouge_metric


def generate_summary(model, tokenizer, text, device="cpu"):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_dialogsum_test(path, n_samples=NUM_EVAL_SAMPLES):
    """Load the first *n_samples* rows from the DialogSum test CSV."""
    dialogues, references = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            dialogues.append(row["dialogue"])
            references.append(row["summary"])
    return dialogues, references


def benchmark_latency(model, tokenizer, text, device="cpu"):
    """Warmup then measure per-run latency, returning a list of durations."""
    for _ in range(NUM_WARMUP):
        generate_summary(model, tokenizer, text, device)

    latencies = []
    for _ in range(NUM_MEASURED):
        start = time.perf_counter()
        generate_summary(model, tokenizer, text, device)
        latencies.append(time.perf_counter() - start)
    return latencies


def benchmark_latency_multi(model, tokenizer, prompts, device="cpu"):
    """Warmup on the first prompt, then time repeated runs on the same prompt (like benchmark_latency)."""
    if not prompts:
        return []
    text = prompts[0]
    for _ in range(NUM_WARMUP):
        generate_summary(model, tokenizer, text, device)

    latencies = []
    for _ in range(NUM_MEASURED):
        start = time.perf_counter()
        generate_summary(model, tokenizer, text, device)
        latencies.append(time.perf_counter() - start)
    return latencies


def generate_predictions_with_timings(model, tokenizer, prompts, device="cpu"):
    """One generation per prompt; returns (predictions, per-sample latencies)."""
    predictions = []
    latencies = []
    for text in prompts:
        start = time.perf_counter()
        pred = generate_summary(model, tokenizer, text, device)
        latencies.append(time.perf_counter() - start)
        predictions.append(pred)
    return predictions, latencies


def compute_eval_metrics(
    model, tokenizer, dialogues, references, device="cpu", predictions=None
):
    """Score summaries against *references*. If *predictions* is set, skips generation."""
    bleu_metric = _get_bleu_metric()
    rouge_metric = _get_rouge_metric()

    if predictions is None:
        predictions = []
        for dialogue in dialogues:
            pred = generate_summary(model, tokenizer, f"summarize: {dialogue}", device)
            predictions.append(pred)
    elif len(predictions) != len(references) or len(predictions) != len(dialogues):
        raise ValueError(
            "predictions, dialogues, and references must have the same length "
            f"({len(predictions)}, {len(dialogues)}, {len(references)})."
        )

    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    return {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
    }


def get_model_param_size_mb(model):
    """Sum of (numel * element_size) for all parameters and buffers."""
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    )
    return total_bytes / (1024 * 1024)


def get_serialized_size_mb(model):
    """Size of the model when serialized to an in-memory buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def get_process_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024 * 1024)


def percentile(data, p):
    """Return the p-th percentile of *data* (0-100)."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def main():
    model_name = "../local_models/t5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    # --- Latency -----------------------------------------------------------
    test_text = (
        "summarize: Speaker1: Hello, how are you? Speaker2: I'm doing well, thank you. "
        "Could you please summarize our conversation?"
    )
    latencies = benchmark_latency(model, tokenizer, test_text, device)

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = percentile(latencies, 95)
    p99_lat = percentile(latencies, 99)
    throughput = 1.0 / mean_lat if mean_lat > 0 else float("inf")

    print("\n=== Latency Benchmark ===")
    print(f"  Warmup runs:  {NUM_WARMUP}")
    print(f"  Measured runs: {NUM_MEASURED}")
    print(f"  Mean latency:  {mean_lat:.4f}s")
    print(f"  Median (p50):  {median_lat:.4f}s")
    print(f"  p95 latency:   {p95_lat:.4f}s")
    print(f"  p99 latency:   {p99_lat:.4f}s")
    print(f"  Throughput:    {throughput:.2f} inferences/sec")

    # --- Evaluation metrics ------------------------------------------------
    test_csv = os.path.join(os.path.dirname(__file__), "..", "local_datasets", "dialogsum", "test.csv")
    test_csv = os.path.normpath(test_csv)

    if os.path.isfile(test_csv):
        print(f"\n=== Evaluation Metrics (on {NUM_EVAL_SAMPLES} DialogSum test samples) ===")
        dialogues, references = load_dialogsum_test(test_csv, NUM_EVAL_SAMPLES)
        metrics = compute_eval_metrics(model, tokenizer, dialogues, references, device)
        print(f"  BLEU:    {metrics['bleu']:.4f}")
        print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    else:
        print(f"\n[WARN] DialogSum test set not found at {test_csv} -- skipping eval metrics.")

    # --- Resource usage ----------------------------------------------------
    param_mb = get_model_param_size_mb(model)
    serial_mb = get_serialized_size_mb(model)
    rss_mb = get_process_rss_mb()
    gpu_mb = get_gpu_memory_mb()

    print("\n=== Resource Usage ===")
    print(f"  Model param size:      {param_mb:.2f} MB")
    print(f"  Serialized model size: {serial_mb:.2f} MB")
    print(f"  Process RSS:           {rss_mb:.2f} MB")
    print(f"  GPU memory allocated:  {f'{gpu_mb:.2f} MB' if gpu_mb is not None else 'N/A (no CUDA)'}")

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
