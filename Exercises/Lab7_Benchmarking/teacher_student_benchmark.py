"""
Teacher-Student Benchmark Comparison

Pipeline: flan-t5-xl (teacher) -> flan-t5-small (student)
  1. KL-divergence distillation (or load checkpoint)
  2. L1 unstructured pruning (20%)
  3. int8 dynamic quantization
  4. Benchmark comparison: size, latency, output

Reuses functions from Lab4 (distillation), Lab5 (pruning),
Lab6 (quantization), and Lab7 (benchmarking).

Requirements: transformers, torch, datasets, psutil, peft, evaluate
"""
import gc
import importlib.util
import os
import statistics

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from datasets import Dataset, load_from_disk
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Load existing lab modules by path (avoids starter_code.py name collisions)
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_lab(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EXERCISES = os.path.dirname(_DIR)

lab4 = _load_lab("lab4_distillation", os.path.join(_EXERCISES, "Lab4_Distillation_TrainingLoop", "starter_code.py"))
lab5 = _load_lab("lab5_pruning", os.path.join(_EXERCISES, "Lab5_Pruning", "starter_code.py"))
lab6 = _load_lab("lab6_quantization", os.path.join(_EXERCISES, "Lab6_Quantization", "starter_code.py"))
lab7 = _load_lab("lab7_benchmark", os.path.join(_DIR, "benchmark.py"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEACHER_LOCAL = os.path.join(_EXERCISES, "local_models", "flan-t5-xl")
STUDENT_LOCAL = os.path.join(_EXERCISES, "local_models", "flan-t5-small")
STUDENT_HF = "google/flan-t5-small"
CHECKPOINT_DIR = os.path.join(_DIR, "distilled_student")
DATASET_DIR = os.path.join(_EXERCISES, "local_datasets", "dialogsum")

NUM_TRAIN_SAMPLES = 100
BATCH_SIZE = 4
LR = 3e-5
NUM_EPOCHS = 1
PRUNE_AMOUNT = 0.1

TEST_CSV = os.path.normpath(os.path.join(DATASET_DIR, "test.csv"))
NUM_BENCHMARK_SAMPLES = 10
NUM_METRICS_SAMPLES = 4
_FALLBACK_PROMPT = (
    "summarize: Speaker1: Hello, how are you? "
    "Speaker2: I'm doing well, thank you. "
    "Could you please summarize our conversation?"
)


def _load_benchmark_data():
    """Load DialogSum test rows for benchmarking (dialogues, references, prompts, has_dataset)."""
    if not os.path.isfile(TEST_CSV):
        print(
            f"[WARN] DialogSum test not found at {TEST_CSV} -- "
            "using a single fallback prompt; BLEU/ROUGE skipped."
        )
        return [], [], [_FALLBACK_PROMPT], False
    dialogues, references = lab7.load_dialogsum_test(TEST_CSV, NUM_BENCHMARK_SAMPLES)
    prompts = [f"summarize: {d}" for d in dialogues]
    return dialogues, references, prompts, True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_teacher(device):
    print(f"Loading teacher from {TEACHER_LOCAL} ...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_LOCAL, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TEACHER_LOCAL, local_files_only=True
    ).to(device)
    model.eval()
    return tokenizer, model


def load_student(device):
    """Load flan-t5-small; downloads from HuggingFace on first run."""
    if os.path.isdir(STUDENT_LOCAL):
        print(f"Loading student from local cache: {STUDENT_LOCAL}")
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_LOCAL, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            STUDENT_LOCAL, local_files_only=True
        )
    else:
        print(f"Downloading student model {STUDENT_HF} ...")
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_HF)
        model = AutoModelForSeq2SeqLM.from_pretrained(STUDENT_HF)
        tokenizer.save_pretrained(STUDENT_LOCAL)
        model.save_pretrained(STUDENT_LOCAL)
        print(f"  Saved to {STUDENT_LOCAL}")
    return tokenizer, model.to(device)


def _load_train_data():
    try:
        ds = load_from_disk(DATASET_DIR)
        return ds.shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    except Exception:
        csv_path = os.path.join(DATASET_DIR, "train.csv")
        print(f"  Falling back to CSV: {csv_path}")
        ds = Dataset.from_csv(csv_path)
        n = min(NUM_TRAIN_SAMPLES, len(ds))
        return ds.shuffle(seed=42).select(range(n))


# ---------------------------------------------------------------------------
# Stage 1 -- Distillation  (reuses lab4.tokenize_function)
# ---------------------------------------------------------------------------
def distill_or_load(teacher_model, teacher_tok, student_model, student_tok, device):
    if os.path.isdir(CHECKPOINT_DIR) and os.path.exists(
        os.path.join(CHECKPOINT_DIR, "config.json")
    ):
        print(f"\nCheckpoint found -- loading distilled student from {CHECKPOINT_DIR}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CHECKPOINT_DIR, local_files_only=True
        ).to(device)
        model.eval()
        return model

    print("\n========== Stage 1: KL-Divergence Distillation ==========")
    train_data = _load_train_data()

    tokenized = [lab4.tokenize_function(ex, teacher_tok) for ex in train_data]

    student_model.train()
    optimizer = AdamW(student_model.parameters(), lr=LR)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"  Student parameters: {total_params:,}")
    print(f"  Training on {len(tokenized)} samples, {NUM_EPOCHS} epoch(s) ...")

    for epoch in range(NUM_EPOCHS):
        for i in range(0, len(tokenized), BATCH_SIZE):
            batch = tokenized[i : i + BATCH_SIZE]
            input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
            attn_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)

            with torch.no_grad():
                t_logits = teacher_model(
                    input_ids=input_ids, attention_mask=attn_mask, labels=labels
                ).logits

            s_logits = student_model(
                input_ids=input_ids, attention_mask=attn_mask, labels=labels
            ).logits

            loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_num = i // BATCH_SIZE + 1
            if batch_num % 5 == 1:
                print(f"  Epoch {epoch + 1}, Batch {batch_num}, Loss: {loss.item():.4f}")

    student_model.eval()
    student_model.save_pretrained(CHECKPOINT_DIR)
    student_tok.save_pretrained(CHECKPOINT_DIR)
    print(f"  Distilled student saved to {CHECKPOINT_DIR}")
    return student_model


# ---------------------------------------------------------------------------
# Stage 2 -- Pruning  (reuses lab5.global_sparsity, lab5.model_size_mb)
# ---------------------------------------------------------------------------
def apply_pruning(model):
    print("\n========== Stage 2: L1 Unstructured Pruning (20%) ==========")
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=PRUNE_AMOUNT)
            prune.remove(module, "weight")

    sparsity = lab5.global_sparsity(model)
    nz_mb = lab5.model_size_mb(model)
    print(f"  Non-zero model size: {nz_mb:.2f} MB")
    return model, sparsity


# ---------------------------------------------------------------------------
# Stage 3 -- Quantization  (reuses lab6.quantize_model)
# ---------------------------------------------------------------------------
def apply_quantization(model):
    print("\n========== Stage 3: int8 Dynamic Quantization ==========")
    return lab6.quantize_model(model, "bfloat16")


# ---------------------------------------------------------------------------
# Benchmarking  (reuses lab7.benchmark_latency, generate_summary, etc.
#                and lab6.get_param_size_mb for quantized models)
# ---------------------------------------------------------------------------
def measure_model(
    label,
    model,
    tokenizer,
    device,
    prompts,
    dialogues,
    references,
    is_quantized=False,
    max_metrics_samples=None,
):
    print(f"  Measuring {label} ...")

    # int8 quantized models live on CPU regardless of device flag
    effective_device = "cpu" if is_quantized else device

    if max_metrics_samples is None:
        max_metrics_samples = NUM_METRICS_SAMPLES

    if is_quantized:
        size_mb = lab6.get_param_size_mb(model)
    else:
        size_mb = lab7.get_model_param_size_mb(model)

    latencies = lab7.benchmark_latency_multi(
        model, tokenizer, prompts, effective_device
    )
    if latencies:
        mean_lat = statistics.mean(latencies)
        p95_lat = lab7.percentile(latencies, 95)
        throughput = 1.0 / mean_lat if mean_lat > 0 else float("inf")
    else:
        mean_lat = 0.0
        p95_lat = 0.0
        throughput = 0.0

    sample_out = ""
    metrics = None
    if prompts:
        if dialogues and references and len(dialogues) == len(references):
            n = min(max_metrics_samples, len(prompts), len(dialogues))
            sub_prompts = prompts[:n]
            sub_dialogues = dialogues[:n]
            sub_refs = references[:n]
            predictions, _ = lab7.generate_predictions_with_timings(
                model, tokenizer, sub_prompts, effective_device
            )
            sample_out = predictions[0] if predictions else ""
            metrics = lab7.compute_eval_metrics(
                model,
                tokenizer,
                sub_dialogues,
                sub_refs,
                effective_device,
                predictions=predictions,
            )
        else:
            sample_out = lab7.generate_summary(
                model, tokenizer, prompts[0], effective_device
            )

    return {
        "label": label,
        "size_mb": size_mb,
        "mean_lat": mean_lat,
        "p95_lat": p95_lat,
        "throughput": throughput,
        "output": sample_out,
        "metrics": metrics,
    }


def print_comparison(results, sparsity):
    col = 22
    sep = "=" * (16 + col * len(results))
    dash = "-" * (16 + col * len(results))
    has_metrics = any(r.get("metrics") for r in results)

    print(f"\n{sep}")
    print("MODEL COMPARISON")
    print(sep)

    print(f"{'':16}" + "".join(r["label"].center(col) for r in results))
    print(dash)

    print(f"{'Size (MB)':16}" + "".join(f"{r['size_mb']:.2f}".center(col) for r in results))

    sparsity_vals = ["--", "--", f"{sparsity:.1f}%"]
    print(f"{'Sparsity':16}" + "".join(v.center(col) for v in sparsity_vals))

    print(f"{'Mean lat (s)':16}" + "".join(f"{r['mean_lat']:.4f}".center(col) for r in results))
    print(f"{'P95 lat (s)':16}" + "".join(f"{r['p95_lat']:.4f}".center(col) for r in results))
    print(f"{'Throughput':16}" + "".join(f"{r['throughput']:.2f} inf/s".center(col) for r in results))

    if has_metrics:
        print(dash)

        def fmt_bleu(r):
            m = r.get("metrics")
            return f"{m['bleu']:.4f}".center(col) if m else "--".center(col)

        def fmt_r1(r):
            m = r.get("metrics")
            return f"{m['rouge1']:.4f}".center(col) if m else "--".center(col)

        def fmt_r2(r):
            m = r.get("metrics")
            return f"{m['rouge2']:.4f}".center(col) if m else "--".center(col)

        def fmt_rl(r):
            m = r.get("metrics")
            return f"{m['rougeL']:.4f}".center(col) if m else "--".center(col)

        print(f"{'BLEU':16}" + "".join(fmt_bleu(r) for r in results))
        print(f"{'ROUGE-1':16}" + "".join(fmt_r1(r) for r in results))
        print(f"{'ROUGE-2':16}" + "".join(fmt_r2(r) for r in results))
        print(f"{'ROUGE-L':16}" + "".join(fmt_rl(r) for r in results))

    print(dash)
    print("Sample output (first test dialogue):")
    for r in results:
        print(f"  {r['label']:12}: {r['output']}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    teacher_tok, teacher_model = load_teacher(device)
    student_tok, student_model = load_student(device)

    # Stage 1
    distilled = distill_or_load(
        teacher_model, teacher_tok, student_model, student_tok, device
    )

    # Benchmark teacher and distilled student BEFORE modifying the student
    print("\n========== Benchmarking ==========")

    dialogues, references, prompts, bench_ok = _load_benchmark_data()
    if bench_ok:
        print(
            f"Using {len(prompts)} DialogSum test samples from {TEST_CSV} "
            f"(max {NUM_BENCHMARK_SAMPLES}); BLEU/ROUGE on first "
            f"{min(NUM_METRICS_SAMPLES, len(prompts))} samples."
        )
        ref0 = references[0] if references else ""
        print(f"Reference (1st sample): {ref0[:300]}{'...' if len(ref0) > 300 else ''}\n")

    first = prompts[0] if prompts else _FALLBACK_PROMPT

    output = lab7.generate_summary(teacher_model, teacher_tok, first, device)
    print(f"Teacher output: {output}")

    output = lab7.generate_summary(distilled, student_tok, first, device)
    print(f"Distilled student output: {output}")

    r_teacher = measure_model(
        "Teacher", teacher_model, teacher_tok, device, prompts, dialogues, references
    )

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    r_student = measure_model(
        "Student", distilled, student_tok, device, prompts, dialogues, references
    )

    # Stage 2
    pruned, sparsity = apply_pruning(distilled)

    output = lab7.generate_summary(pruned, student_tok, first, device)
    print(f"Pruned student output: {output}")

    # Stage 3  (moves model to CPU for int8)
    final = apply_quantization(pruned)

    r_final = measure_model(
        "Final",
        final,
        student_tok,
        device,
        prompts,
        dialogues,
        references,
        is_quantized=True,
    )

    print_comparison([r_teacher, r_student, r_final], sparsity)


if __name__ == "__main__":
    main()
