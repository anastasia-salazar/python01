"""LoRA vs QLoRA fine-tuning of a 1B LLaMA-based instruction model.

Fine-tunes TinyLlama-1.1B-Chat (LLaMA architecture) on a small instruction dataset twice:
  1. LoRA:  base model frozen, rank-8 adapters on the attention layers q_proj and v_proj
  2. QLoRA: same, but the frozen base model is loaded in 4-bit NF4 with bitsandbytes
Both use identical settings (r=8, 2 epochs, learning rate 2e-4). For each we record trainable
parameters, final training loss and peak memory, then compare answers to 3 test prompts.

Runs on an NVIDIA GPU if one is available, otherwise on the CPU (slower, so fewer
training examples are used by default). Peak memory is GPU memory on a GPU, RAM on a CPU.

Each experiment runs in its own process so its peak memory is measured cleanly, and its
results are saved to results/ as soon as it finishes. Re-running skips finished experiments.

Usage:
    python llm4.py                  # run everything, then write the report
    python llm4.py --fresh          # ignore saved results and rerun everything
    python llm4.py --samples 100    # use a different number of training examples
    python llm4.py --report         # only rebuild the report from saved results

Requirements:
    pip install -U torch transformers peft bitsandbytes accelerate datasets pandas matplotlib
"""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# ---- Experiment settings (identical for LoRA and QLoRA) ----
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Alternative: "meta-llama/Llama-3.2-1B-Instruct" (needs a Hugging Face token and
# accepting Meta's license on the model page first)

DATASET_NAME = "yahma/alpaca-cleaned"  # instruction / input / output examples
GPU_SAMPLES = 1000                     # training examples on a GPU
CPU_SAMPLES = 200                      # fewer on a CPU, which is much slower
MAX_LENGTH = 256                       # max tokens per example

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # attention layers only

EPOCHS = 2
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4                  # effective batch size = 4 x 4 = 16
SEED = 42

TEST_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Give three tips for staying productive while working from home.",
    "Write a short, polite email declining a meeting invitation.",
]
MAX_NEW_TOKENS = 200

METHODS = ["base", "lora", "qlora"]
DISPLAY_NAMES = {"base": "Base model", "lora": "LoRA", "qlora": "QLoRA"}


# =====================================================================
# Part 1: one experiment (runs in its own process)
# =====================================================================
def run_experiment(method, num_samples):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    torch.manual_seed(SEED)
    on_gpu = torch.cuda.is_available()
    if on_gpu:
        device = "GPU: " + torch.cuda.get_device_name(0)
        # T4 GPUs don't support bfloat16, so use float16 there
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        device_map = "auto"
    else:
        # PyTorch defaults to physical cores only; using every CPU thread is usually faster
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        device = f"CPU ({torch.get_num_threads()} threads)"
        # Most CPUs have no fast 16-bit math, so float32 is faster there
        use_bf16 = False
        dtype = torch.float32
        device_map = "cpu"
    print(f"\n=== {DISPLAY_NAMES[method]} on {device}, dtype {dtype} ===", flush=True)

    ram_before_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    # ---- Tokenizer and prompt format ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def build_prompt(instruction):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(model, prompt):
        """Greedy decoding, so outputs are deterministic and comparable between models."""
        model.eval()
        inputs = tokenizer(build_prompt(prompt), return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ---- Load the model ----
    if method == "qlora":
        # Frozen base weights stored in 4-bit NormalFloat4; double quantization also
        # compresses the quantization constants. Computation happens in `dtype`.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config,
                                                     device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype,
                                                     device_map=device_map)
    weights_gb = model.get_memory_footprint() / 1e9
    print(f"Model weights in memory: {weights_gb:.2f} GB", flush=True)

    result = {"method": method, "device": device, "dtype": str(dtype).replace("torch.", ""),
              "weights_gb": weights_gb}

    if method != "base":
        # ---- Training data: loss only on the response, not the prompt ----
        raw = load_dataset(DATASET_NAME, split="train").shuffle(seed=SEED).select(range(num_samples))

        def tokenize_example(example):
            instruction = example["instruction"]
            if example["input"]:
                instruction += "\n\n" + example["input"]
            prompt_text = build_prompt(instruction)
            full_text = prompt_text + example["output"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full = tokenizer(full_text, add_special_tokens=False, truncation=True,
                             max_length=MAX_LENGTH)
            # Prompt tokens get label -100, so they are ignored by the loss
            labels = [-100] * len(prompt_ids) + full["input_ids"][len(prompt_ids):]
            full["labels"] = labels[:len(full["input_ids"])]
            return full

        train_data = raw.map(tokenize_example, remove_columns=raw.column_names)
        # Drop examples whose response was cut off entirely by MAX_LENGTH
        train_data = train_data.filter(lambda ex: any(label != -100 for label in ex["labels"]))

        # ---- Add LoRA adapters ----
        if method == "qlora":
            # Prepares a quantized model for training (e.g. keeps layer norms in float32).
            # Gradient checkpointing stays off so the only difference from LoRA is the 4-bit base.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
        ))
        # Keep the small trainable adapters in float32 for stable mixed-precision training
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        trainable, total = model.get_nb_trainable_parameters()
        base_frozen = all(not p.requires_grad
                          for name, p in model.named_parameters() if "lora_" not in name)
        adapted = sorted({name.split(".")[-1] for name, module in model.named_modules()
                          if hasattr(module, "lora_A") and name.split(".")[-1] in TARGET_MODULES})
        print(f"Trainable parameters: {trainable:,} of {total:,} "
              f"({100 * trainable / total:.3f}%)")
        print(f"Base model weights frozen: {base_frozen}")
        print(f"LoRA adapters added to: {adapted}", flush=True)

        # ---- Train ----
        steps = math.ceil(len(train_data) / (BATCH_SIZE * GRAD_ACCUMULATION)) * EPOCHS
        args = TrainingArguments(
            output_dir=str(HERE / "outputs" / method),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            warmup_steps=max(1, steps // 20),
            logging_steps=max(1, steps // 15),
            save_strategy="no",
            report_to="none",
            fp16=on_gpu and not use_bf16,
            bf16=on_gpu and use_bf16,
            use_cpu=not on_gpu,
            seed=SEED,
        )
        collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
        trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=collator)

        print(f"Training on {len(train_data)} examples for {EPOCHS} epochs ({steps} steps)...",
              flush=True)
        start = time.time()
        train_output = trainer.train()
        step_logs = [log for log in trainer.state.log_history if "loss" in log]

        result.update(
            trainable_params=trainable,
            total_params=total,
            base_frozen=base_frozen,
            adapted_modules=adapted,
            train_examples=len(train_data),
            train_minutes=(time.time() - start) / 60,
            final_loss=step_logs[-1]["loss"],        # loss at the last logged step
            average_loss=train_output.training_loss,  # average over the whole run
            loss_curve=[[log["step"], log["loss"]] for log in step_logs],
        )
        model.save_pretrained(str(HERE / "adapters" / method))  # only the small adapters

    # Peak memory for this whole process: loading + training
    if on_gpu:
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        result["memory_type"] = "GPU memory"
    else:
        result["peak_memory_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        result["memory_type"] = "RAM"
    result["memory_before_model_gb"] = ram_before_gb if not on_gpu else 0.0

    print("Generating test prompt answers...", flush=True)
    result["outputs"] = [generate(model, prompt) for prompt in TEST_PROMPTS]

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{method}.json").write_text(json.dumps(result, indent=2))
    if method != "base":
        print(f"Final training loss: {result['final_loss']:.4f}")
    print(f"Peak {result['memory_type']}: {result['peak_memory_gb']:.2f} GB")


# =====================================================================
# Part 2: the report comparing all runs
# =====================================================================
def write_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {m: json.loads((RESULTS_DIR / f"{m}.json").read_text())
            for m in METHODS if (RESULTS_DIR / f"{m}.json").exists()}
    if not {"lora", "qlora"} <= runs.keys():
        print("LoRA and QLoRA results are both needed for the report.")
        return
    lora, qlora = runs["lora"], runs["qlora"]
    memory_type = lora["memory_type"]

    rows = [
        ("Base weights precision", lora["dtype"], "4-bit NF4"),
        ("Trainable parameters", f"{lora['trainable_params']:,}", f"{qlora['trainable_params']:,}"),
        ("% of all parameters",
         f"{100 * lora['trainable_params'] / lora['total_params']:.3f}%",
         f"{100 * qlora['trainable_params'] / qlora['total_params']:.3f}%"),
        ("Model weights (GB)", f"{lora['weights_gb']:.2f}", f"{qlora['weights_gb']:.2f}"),
        (f"Peak {memory_type} (GB)", f"{lora['peak_memory_gb']:.2f}",
         f"{qlora['peak_memory_gb']:.2f}"),
        ("Final training loss", f"{lora['final_loss']:.4f}", f"{qlora['final_loss']:.4f}"),
        ("Average training loss", f"{lora['average_loss']:.4f}", f"{qlora['average_loss']:.4f}"),
        ("Training time (min)", f"{lora['train_minutes']:.1f}", f"{qlora['train_minutes']:.1f}"),
    ]
    saving = 100 * (1 - qlora["peak_memory_gb"] / lora["peak_memory_gb"])
    weight_saving = 100 * (1 - qlora["weights_gb"] / lora["weights_gb"])
    loss_gap = qlora["final_loss"] - lora["final_loss"]

    # ---- Print the table ----
    print("\n" + "=" * 70)
    print(f"{'':28s}{'LoRA':>20s}{'QLoRA':>20s}")
    for label, a, b in rows:
        print(f"{label:28s}{a:>20s}{b:>20s}")
    print(f"\nQLoRA used {saving:.0f}% less peak {memory_type} and {weight_saving:.0f}% less "
          f"memory for model weights than LoRA.")
    print(f"Final loss difference (QLoRA - LoRA): {loss_gap:+.4f}")

    # ---- Loss curve chart ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("lora", "#2a78d6"), ("qlora", "#eb6834")]:
        steps, losses = zip(*runs[name]["loss_curve"])
        ax.plot(steps, losses, label=DISPLAY_NAMES[name], color=color, linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss: LoRA vs QLoRA")
    ax.legend(frameon=False)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ---- Markdown report with a section for the qualitative comparison ----
    lines = [
        "# LoRA vs QLoRA Fine-Tuning Report",
        "",
        f"- **Model:** {MODEL_NAME}",
        f"- **Hardware:** {lora['device']}",
        f"- **Data:** {lora['train_examples']} examples from {DATASET_NAME}",
        f"- **Settings (both runs):** r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, "
        f"target modules={TARGET_MODULES}, {EPOCHS} epochs, learning rate {LEARNING_RATE}, "
        f"effective batch size {BATCH_SIZE * GRAD_ACCUMULATION}",
        f"- **Base weights frozen:** LoRA {lora['base_frozen']}, QLoRA {qlora['base_frozen']}",
        "",
        "## Results",
        "",
        "| Metric | LoRA | QLoRA |",
        "|---|---|---|",
        *[f"| {label} | {a} | {b} |" for label, a, b in rows],
        "",
        f"QLoRA used **{saving:.0f}% less peak {memory_type}** and **{weight_saving:.0f}% less "
        f"memory for model weights**, with a final loss difference of **{loss_gap:+.4f}**.",
        "",
        "![Training loss](loss_curve.png)",
        "",
        "## Test prompt outputs",
    ]
    for i, prompt in enumerate(TEST_PROMPTS):
        lines += ["", f"### Prompt {i + 1}: {prompt}"]
        for method in METHODS:
            if method in runs:
                answer = runs[method]["outputs"][i].replace("\n", "\n> ")
                lines += ["", f"**{DISPLAY_NAMES[method]}:**", "", f"> {answer}"]
    lines += [
        "",
        "## Qualitative comparison",
        "",
        "*Fill this in after reading the outputs above. For each prompt, compare instruction "
        "following (e.g. exactly three tips?), accuracy, clarity/structure, and whether LoRA "
        "and QLoRA answers differ noticeably in quality.*",
        "",
        "| Prompt | Base model | LoRA | QLoRA |",
        "|---|---|---|---|",
        "| 1. List vs tuple | | | |",
        "| 2. Productivity tips | | | |",
        "| 3. Declining email | | | |",
        "",
        "**Conclusion:** ",
        "",
    ]
    report = RESULTS_DIR / "report.md"
    report.write_text("\n".join(lines))
    print(f"\nReport saved to {report}")
    print(f"Loss chart saved to {RESULTS_DIR / 'loss_curve.png'}")


# =====================================================================
# Part 3: run each experiment in its own process, then report
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)  # used internally
    parser.add_argument("--samples", type=int, help="number of training examples")
    parser.add_argument("--fresh", action="store_true", help="rerun experiments with saved results")
    parser.add_argument("--report", action="store_true", help="only rebuild the report")
    args = parser.parse_args()

    if args.method:
        run_experiment(args.method, args.samples)
        return 0
    if args.report:
        write_report()
        return 0

    import torch
    samples = args.samples or (GPU_SAMPLES if torch.cuda.is_available() else CPU_SAMPLES)

    for method in METHODS:
        if (RESULTS_DIR / f"{method}.json").exists() and not args.fresh:
            print(f"Skipping {DISPLAY_NAMES[method]}: results already saved "
                  f"(use --fresh to rerun)")
            continue
        command = [sys.executable, __file__, "--method", method, "--samples", str(samples)]
        if subprocess.run(command).returncode != 0:
            print(f"\n{DISPLAY_NAMES[method]} failed; see the error above. "
                  f"Re-run the same command to continue from here.")
            return 1

    write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
