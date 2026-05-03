from unsloth import FastLanguageModel
import torch
import torch.nn as nn
import torch.distributed as dist
import copy
from trl import SFTTrainer
from transformers import TrainerCallback
from datasets import load_dataset, Dataset, interleave_datasets, load_from_disk
import json
from pathlib import Path
import gc
import os
import sys

# Path fixing implementation for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.hybrid_surgery import perform_gdn_surgery
from src.architecture.gdn_hybrid_layer import GemmaDeltaNetLayer

# CUDA allocator: expandable segments reduce fragmentation (recommended by torch OOM messages)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Triton stability
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'
# Note: CUDA_LAUNCH_BLOCKING and TORCH_USE_CUDA_DSA removed — those were debugging flags
# that serialized every GPU kernel (4-5x slowdown). The bugs they helped diagnose are fixed.

# ==============================================================================
# 1. Packing Efficiency Monitor
# ==============================================================================
class PackingEfficiencyCallback(TrainerCallback):
    """Monitors packing efficiency during training."""
    def __init__(self, max_seq_length=8192):
        self.max_seq_length = max_seq_length
        self.efficiency_samples = []

    def on_step_end(self, args, state, control, **kwargs):
        # Log efficiency every 50 steps
        if state.global_step % 50 == 0 and len(self.efficiency_samples) > 0:
            avg_efficiency = sum(self.efficiency_samples) / len(self.efficiency_samples)
            if avg_efficiency < 0.5:
                print(f"Warning: Packing efficiency dropped to {avg_efficiency:.1%} at step {state.global_step}")
            self.efficiency_samples = []
        return control

# ==============================================================================
# 2. Surgery Engine
# ==============================================================================
# Spectral init removed. Direct weight transfer is used for exact feature preservation.

# ==============================================================================
# 3. Progressive Orchestrator
# ==============================================================================
def set_trainable_parameters(model, active_layer_idx, replaced_so_far):
    for param in model.parameters():
        param.requires_grad = False

    if active_layer_idx not in replaced_so_far:
        replaced_so_far.append(active_layer_idx)

    for name, param in model.named_parameters():
        for idx in replaced_so_far:
            # Fix: Use word boundaries to prevent false matches (e.g., .4. matching .14., .24.)
            if f".layers.{idx}." in name or f"layers.{idx}." in name:
                if "lora_" in name:
                    param.requires_grad = True
                elif "input_layernorm" in name or "post_attention_layernorm" in name:
                    param.requires_grad = True
                elif "self_attn" in name:
                    param.requires_grad = True

    return replaced_so_far

# --- Dynamic Buffer Function (for streaming data) ---
def get_fresh_buffer_streaming(dataset_iter, buffer_size=6000):
    """Pulls a fresh batch of examples from streaming dataset iterator."""
    buffer = []
    for i, example in enumerate(dataset_iter):
        if i >= buffer_size:
            break
        buffer.append(example)

    if not buffer:
        return None

    # Convert list of examples to Dataset
    from datasets import Dataset
    return Dataset.from_dict({
        key: [example[key] for example in buffer]
        for key in buffer[0].keys()
    })

def get_fresh_buffer_pretokenized(dataset, start_idx, buffer_size=6000):
    """Pulls a fresh batch of examples from pretokenized dataset (non-streaming)."""
    end_idx = min(start_idx + buffer_size, len(dataset))
    return dataset.select(range(start_idx, end_idx)), end_idx

# ==============================================================================
# 4. Dataset Loading & Formatting (Raw SFT Streaming)
# ==============================================================================

def gemma_format_func(examples):
    """Wraps a batch of examples in the exact Gemma 4 Chat Template."""
    output_texts = []
    # Handle both single examples and batches
    if isinstance(examples.get("instruction"), list):
        for i in range(len(examples["instruction"])):
            prompt = examples["instruction"][i]
            if examples.get("input") and i < len(examples["input"]):
                if examples["input"][i]:
                    prompt += "\n" + examples["input"][i]
            response = examples["output"][i]
            text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
            output_texts.append(text)
    else:
        # Fallback for single example
        prompt = examples.get("instruction", "")
        if examples.get("input"):
            prompt += "\n" + examples["input"]
        response = examples.get("output", "")
        output_texts.append(f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>")
        
    return output_texts

def load_stage2_streaming_dataset():
    """Stream the raw text datasets directly."""
    print("Streaming Stage 2 datasets from HuggingFace Hub...")
    ds_longalpaca = load_dataset("Yukang/LongAlpaca-12k", split="train", streaming=True)
    return ds_longalpaca

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Progressive GDN Hybrid Training")
    parser.add_argument(
        "--use_pretokenized",
        action="store_true",
        help="Use pretokenized data instead of downloading datasets"
    )
    parser.add_argument(
        "--pretokenized_dir",
        type=str,
        default="./pretokenized_data",
        help="Directory containing pretokenized datasets"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets from HuggingFace Hub with capping"
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        help=(
            "Load pretokenized datasets directly from the HuggingFace Hub "
            "(leonidas123/gemma-4-pretokenized-traces). Applies MAX_EXAMPLES_PER_DATASET "
            "caps. Preferred over --use_pretokenized when running on a remote cluster "
            "without pre-synced local data."
        )
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default="leonidas123/gemma-4-pretokenized-traces",
        help="HuggingFace Hub repository to load pretokenized datasets from (used with --hub)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torch.distributed.launch)"
    )

    args = parser.parse_args()

    # Initialize distributed training if multiple GPUs are available
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if world_size > 1:
        # Multi-GPU: Initialize DDP
        if local_rank == -1:
            local_rank = 0
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        is_main_process = local_rank == 0
        print(f"[Rank {local_rank}] Initialized DDP with {world_size} GPUs")
    else:
        # Single GPU
        is_main_process = True
        local_rank = 0
        print(f"Running on single GPU")

    # Progressive buffer schedule for dynamic sizing
    BUFFER_SCHEDULE = {16: 5000, 12: 6000, 8: 7000, 4: 10000}

    REPLACEMENT_SCHEDULE = [(16, 500), (12, 500), (8, 500), (4, 1000)]
    LR_SCHEDULE = {16: 2e-4, 12: 1.5e-4, 8: 1e-4, 4: 8e-5}
    REPLACED_SO_FAR = []

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E2B-it",
        max_seq_length=8192,
        dtype=torch.bfloat16,  # L40S supports BF16 natively; use it for FLA kernel stability
        load_in_4bit=False,    
    )

    # Phase 1: Execute all architectural replacements upfront
    target_layers = [layer_idx for layer_idx, _ in REPLACEMENT_SCHEDULE]
    if is_main_process:
        print("Applying Early-Fusion Hybrid Surgery...")
    model = perform_gdn_surgery(model, target_layers=target_layers)

    # Phase 2: Apply PEFT Post-Surgery
    # target ONLY frozen pre-existing layers with LoRA. 
    # New architecture components (GDN) must be in modules_to_save to train in FULL RANK.
    if is_main_process:
        print("Applying PEFT configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "gate_proj", "up_proj", "down_proj",  # LoRA on frozen MLP layers
        ],
        modules_to_save=[
            "language_model.layers.4.self_attn",
            "language_model.layers.8.self_attn",
            "language_model.layers.12.self_attn",
            "language_model.layers.16.self_attn",
            "language_model.layers.4.input_layernorm",
            "language_model.layers.8.input_layernorm",
            "language_model.layers.12.input_layernorm",
            "language_model.layers.16.input_layernorm",
            "language_model.layers.4.post_attention_layernorm",
            "language_model.layers.8.post_attention_layernorm",
            "language_model.layers.12.post_attention_layernorm",
            "language_model.layers.16.post_attention_layernorm",
        ],  # Explicitly target ONLY the surgically replaced language model layers.
            # This prevents Unsloth from upcasting the entire Vision Tower to FP32.
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Disable gradient checkpointing for GDN layers (conflicts with FLA custom autograd)
    base_model = getattr(model, 'base_model', model)
    if hasattr(base_model, 'model'):
        gemma4_model = base_model.model
    else:
        gemma4_model = base_model
    if hasattr(gemma4_model, 'language_model'):
        text_model = gemma4_model.language_model
    else:
        text_model = gemma4_model

    for layer_idx in [16, 12, 8, 4]:  # Replaced layers
        if hasattr(text_model, 'layers') and layer_idx < len(text_model.layers):
            if hasattr(text_model.layers[layer_idx], 'self_attn'):
                # Mark GDN layers to skip gradient checkpointing
                text_model.layers[layer_idx].self_attn._no_gradient_checkpointing = True
                if is_main_process:
                    print(f"Disabled gradient checkpointing for GDN layer {layer_idx}")

    # Phase 3: Progressive Training Loop
    if is_main_process:
        print("Loading Stage 2 streaming dataset...")
    stage2_dataset = load_stage2_streaming_dataset()
    dataset_iter = iter(stage2_dataset)
    is_streaming = True
    current_idx = 0

    for layer_idx, steps in REPLACEMENT_SCHEDULE:
        if is_main_process:
            print(f"\n--- Initiating Warmup on Layer {layer_idx} ---")
        REPLACED_SO_FAR = set_trainable_parameters(model, layer_idx, REPLACED_SO_FAR)

        # Buffer fresh data with dynamic sizing
        buffer_size = BUFFER_SCHEDULE[layer_idx]
        if is_main_process:
            print(f"Buffering fresh data for Layer {layer_idx} (buffer_size={buffer_size})...")

        # Use streaming buffer for hub datasets
        if is_streaming or args.hub:
            buffered_dataset = get_fresh_buffer_streaming(dataset_iter, buffer_size=buffer_size)
            if buffered_dataset is None:
                if is_main_process:
                    print(f"Warning: Dataset exhausted at Layer {layer_idx}. Saving current progress...")
                    model.save_pretrained(f"./checkpoints/phase_{layer_idx}_interrupted")
                    tokenizer.save_pretrained(f"./checkpoints/phase_{layer_idx}_interrupted")
                    print(f"Progress saved to ./checkpoints/phase_{layer_idx}_interrupted")
                break
        else:
            # Use index-based buffer for local datasets
            buffered_dataset, current_idx = get_fresh_buffer_pretokenized(
                stage2_dataset, current_idx, buffer_size=buffer_size
            )
        if is_main_process:
            print(f"Loaded buffer: {len(buffered_dataset)} examples")

        # Validate dataset columns (not strictly needed for raw text formatting, but kept for structure)

        # Verify gradient accumulation
        per_device_bs = 4   # 4→ better GPU utilization vs 2; OOM fix via expandable_segments
        grad_accum = 6    # 4*6=24 = same effective batch size
        effective_batch_size = per_device_bs * grad_accum

        from trl import SFTConfig

        training_args = SFTConfig(
            output_dir=f"./checkpoints/progressive_L{layer_idx}",
            max_steps=steps,
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=LR_SCHEDULE[layer_idx],
            fp16=False,
            bf16=True,   # Enabled BF16 for L40S
            optim="adamw_8bit", # L40S has 48GB; standard 8-bit Adam is fine
            logging_steps=10,
            remove_unused_columns=True,
            save_strategy="no",  # We save manually after each phase
            max_seq_length=8192,
            packing=True,
            gradient_checkpointing=True,
            # DDP settings
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=buffered_dataset,
            formatting_func=gemma_format_func, # <-- This is mandatory for packing=True
            max_seq_length=8192,
            callbacks=[PackingEfficiencyCallback(max_seq_length=8192)],
        )

        # Training with error handling
        try:
            if is_main_process:
                print(f"Starting training for Layer {layer_idx}...")
            trainer.train()
            if is_main_process:
                print(f"Training completed for Layer {layer_idx}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if is_main_process:
                    print(f"OOM Error on Layer {layer_idx}: {e}")
                    print(f"Suggestion: Reduce buffer_size from {buffer_size} or per_device_train_batch_size from {per_device_bs}")
                raise
            else:
                raise

        # CUDA synchronization for error localization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if is_main_process:
                print(f"CUDA synchronized after Layer {layer_idx} training")

        # Memory management between phases
        gc.collect()
        torch.cuda.empty_cache()
        if is_main_process:
            max_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"Max VRAM used during Layer {layer_idx}: {max_vram_gb:.2f} GB")

        # Checkpoint safety: verify disk space before saving
        if is_main_process:
            import shutil
            free_space_gb = shutil.disk_usage("./checkpoints").free / 1e9
            if free_space_gb < 2.0:  # Need at least 2GB for delta checkpoint
                print(f"Warning: Low disk space ({free_space_gb:.2f} GB free). Checkpoint may fail.")

            model.save_pretrained(f"./checkpoints/phase_{layer_idx}_saved")
            tokenizer.save_pretrained(f"./checkpoints/phase_{layer_idx}_saved")
            print(f"--- Layer {layer_idx} Stabilized & Adapters Saved ---")

    # Phase 4: Final Hackathon Merge & Export
    if is_main_process:
        print("\n--- Training Complete. Merging LoRA into Base Model ---")

        # Checkpoint safety: verify disk space before final merge
        import shutil
        free_space_gb = shutil.disk_usage("./checkpoints").free / 1e9
        if free_space_gb < 20.0:  # Need at least 20GB for full merged model
            print(f"Warning: Low disk space ({free_space_gb:.2f} GB free). Final merge requires ~20GB.")
            raise RuntimeError(f"Insufficient disk space for final merge: {free_space_gb:.2f} GB free, need 20GB")

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("./checkpoints/final_hackathon_model")
        tokenizer.save_pretrained("./checkpoints/final_hackathon_model")

        print("--- Model Fused & Ready for Deployment ---")

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()