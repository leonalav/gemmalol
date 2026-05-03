from unsloth import FastLanguageModel
import torch
import torch.nn as nn
import torch.distributed as dist
import copy
from trl import SFTTrainer
from transformers import TrainerCallback
from datasets import load_dataset, Dataset, interleave_datasets, load_from_disk
from architecture.gdn_hybrid_layer import GemmaDeltaNetLayer
import json
from pathlib import Path
import gc
import os

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

def apply_hybrid_surgery(model, layer_idx):
    # Navigate to text model: Unsloth wraps models, so we need to find the actual Gemma4 structure
    # model could be: PeftModel -> base_model -> Gemma4ForConditionalGeneration -> model (Gemma4Model) -> language_model (Gemma4TextModel)

    # Unwrap PEFT if present (though this runs before PEFT is applied)
    base_model = getattr(model, 'base_model', model)

    # Find Gemma4Model (the multimodal wrapper)
    if hasattr(base_model, 'model'):
        gemma4_model = base_model.model
    else:
        gemma4_model = base_model

    # Find Gemma4TextModel (the actual text backbone with layers)
    if hasattr(gemma4_model, 'language_model'):
        text_model = gemma4_model.language_model
    else:
        text_model = gemma4_model

    if not hasattr(text_model, 'layers'):
        raise AttributeError(f"Could not find 'layers' attribute. Model structure: {type(model)} -> {type(base_model)} -> {type(gemma4_model)} -> {type(text_model)}")

    original_layer = text_model.layers[layer_idx]
    old_attn = original_layer.self_attn

    # Get the correct config (text_config for Gemma4Model, or config for Gemma4TextModel)
    if hasattr(gemma4_model, 'config') and hasattr(gemma4_model.config, 'text_config'):
        layer_config = gemma4_model.config.text_config
    elif hasattr(text_model, 'config'):
        layer_config = text_model.config
    else:
        layer_config = base_model.config.text_config if hasattr(base_model.config, 'text_config') else base_model.config

    new_layer = GemmaDeltaNetLayer(layer_config, layer_idx).to(model.device)

    # Copy frozen components from original layer
    new_layer.mlp = original_layer.mlp
    if hasattr(original_layer, 'router'):
        new_layer.router = original_layer.router
        new_layer.experts = original_layer.experts
        new_layer.post_feedforward_layernorm_1 = original_layer.post_feedforward_layernorm_1
        new_layer.post_feedforward_layernorm_2 = original_layer.post_feedforward_layernorm_2
        new_layer.pre_feedforward_layernorm_2 = original_layer.pre_feedforward_layernorm_2

    new_layer.layer_scalar = original_layer.layer_scalar

    if new_layer.hidden_size_per_layer_input > 0:
        new_layer.per_layer_input_gate = original_layer.per_layer_input_gate
        new_layer.per_layer_projection = original_layer.per_layer_projection
        new_layer.post_per_layer_input_norm = original_layer.post_per_layer_input_norm

    # Initialize GDN projections with pretrained Gemma weights (warm start)
    new_layer.self_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
    new_layer.self_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
    new_layer.self_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
    new_layer.self_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

    text_model.layers[layer_idx] = new_layer
    return model

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
# 4. Dataset Loading & Formatting
# ==============================================================================

# Dataset capping strategy for 12-13 hour training window
# Visual datasets (llava, sharegpt4v, visual_cot) are EXCLUDED from progressive warm-start.
# During warm-start the GDN cross-attention is zero-initialized and dormant — no
# multimodal_states are ever injected into the forward pass. Visual datasets would
# only contribute stripped-text description-style language patterns that dilute
# the instruct/reasoning distribution. They are deferred to Phase 4 multimodal alignment.
MAX_EXAMPLES_PER_DATASET = {
    "long_alpaca": None,              # Use all ~12k
    "maths_grade_school": None,       # Use all
    "education_young_children": None, # Use all
    "megascience": 10_000,            # Cap from larger set
    "medquad": None,                  # Use all ~16k
    "glaive_code_assistant": 10_000,  # Cap from larger set
    "glaive_function_calling": 10_000,# Cap from larger set
    "opus_10k": None,                 # Use all 10k — CoT heavyweight
    "opus_3300": None,                # Use all 3.3k — CoT reasoning
}

# Visual dataset keys to skip during warm-start loading.
# These may be present in metadata.json from a prior pretokenization run that
# included visual data. We filter them at load time rather than regenerating
# metadata so the Hub data remains intact for Phase 4 multimodal alignment.
_VISUAL_DATASET_KEYS = {"llava", "sharegpt4v", "visual_cot"}

def load_stage2_pretokenized_dataset(pretokenized_dir="./pretokenized_data"):
    """Loads pretokenized datasets from disk, applies capping, and interleaves them.
    
    Visual datasets (llava, sharegpt4v, visual_cot) are silently skipped even if
    present in metadata.json — cross-attention is zero-init and dormant during
    warm-start, so they add no signal and only dilute the reasoning distribution.
    """
    print(f"Loading pretokenized datasets from {pretokenized_dir}...")

    pretokenized_path = Path(pretokenized_dir)

    # Load metadata
    metadata_path = pretokenized_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Metadata: {metadata}")

    # Dataset name mapping for capping (visual datasets intentionally absent)
    dataset_name_map = {
        "long_alpaca": "long_alpaca",
        "maths_grade_school": "maths_grade_school",
        "education_young_children": "education_young_children",
        "megascience": "megascience",
        "medquad": "medquad",
        "glaive_code_assistant": "glaive_code_assistant",
        "glaive_function_calling": "glaive_function_calling",
        "opus_10k": "opus_10k",
        "opus_3300": "opus_3300",
    }

    # Load each pretokenized dataset with capping, skipping visual datasets
    datasets = []
    for dataset_path in metadata["datasets"]:
        # Skip visual datasets — deferred to Phase 4 multimodal alignment
        is_visual = any(key in dataset_path.lower() for key in _VISUAL_DATASET_KEYS)
        if is_visual:
            print(f"  [SKIP] {dataset_path} — visual dataset excluded from warm-start")
            continue

        print(f"Loading {dataset_path}...")
        ds = load_from_disk(dataset_path)

        # Identify dataset name from path
        dataset_name = None
        for key in dataset_name_map.keys():
            if key in dataset_path.lower():
                dataset_name = key
                break

        # Apply capping if specified
        if dataset_name and MAX_EXAMPLES_PER_DATASET.get(dataset_name) is not None:
            cap = MAX_EXAMPLES_PER_DATASET[dataset_name]
            if len(ds) > cap:
                print(f"  Capping {dataset_name} from {len(ds)} to {cap} examples")
                ds = ds.shuffle(seed=42).select(range(cap))

        datasets.append(ds)
        print(f"  Loaded {len(ds)} examples")

    # Interleave datasets with equal probability
    if len(datasets) > 1:
        interleaved = interleave_datasets(
            datasets,
            probabilities=[1.0 / len(datasets)] * len(datasets),
            stopping_strategy="all_exhausted"
        )
    else:
        interleaved = datasets[0]

    print(f"Total interleaved dataset size: {len(interleaved)}")
    return interleaved

def load_stage2_downloaded_dataset():
    """Downloads datasets from HuggingFace Hub with capping applied during download."""
    print("Downloading Stage 2 datasets from HuggingFace Hub with capping...")

    datasets = []

    # Load each dataset with capping
    # Visual datasets (LLaVA, ShareGPT4V, Visual-CoT) are excluded from warm-start.
    # During this phase the GDN cross-attention is zero-initialized and dormant;
    # multimodal_states are never injected. Visual data only adds description-style
    # noise. Defer to Phase 4 multimodal alignment with active cross-attention.
    dataset_configs = [
        ("Yukang/LongAlpaca-12k", None, "long_alpaca"),
        ("ajibawa-2023/Maths-Grade-School", None, "maths_grade_school"),
        ("ajibawa-2023/Education-Young-Children", None, "education_young_children"),
        ("MegaScience/MegaScience", None, "megascience"),
        ("keivalya/MedQuad-MedicalQnADataset", None, "medquad"),
        ("glaiveai/glaive-code-assistant-v3", None, "glaive_code_assistant"),
        ("glaiveai/glaive-function-calling-v2", None, "glaive_function_calling"),
        ("Roman1111111/claude-opus-4.6-10000x", None, "opus_10k"),
        ("Crownelius/Opus-4.6-Reasoning-3300x", None, "opus_3300"),
    ]

    for dataset_name, config, key in dataset_configs:
        print(f"Downloading {dataset_name}...")

        cap = MAX_EXAMPLES_PER_DATASET.get(key)

        if config:
            ds = load_dataset(dataset_name, config, split="train", streaming=False)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=False)

        # Apply capping with reproducible sampling
        if cap is not None and len(ds) > cap:
            print(f"  Capping {key} from {len(ds)} to {cap} examples")
            ds = ds.shuffle(seed=42).select(range(cap))

        datasets.append(ds)
        print(f"  Loaded {len(ds)} examples")

    # Interleave datasets with equal probability
    interleaved = interleave_datasets(
        datasets,
        probabilities=[1.0 / len(datasets)] * len(datasets),
        stopping_strategy="all_exhausted"
    )

    print(f"Total interleaved dataset size: {len(interleaved)}")
    return interleaved

def load_stage2_from_hub(hf_repo="leonidas123/gemma-4-pretokenized-traces"):
    """
    Streams pretokenized Stage 2 datasets directly from the HuggingFace Hub,
    applying the caps defined in MAX_EXAMPLES_PER_DATASET.

    Uses streaming=True to avoid downloading full datasets to disk.
    """
    print(f"Streaming Stage 2 datasets from HuggingFace Hub: {hf_repo}")

    HUB_DATASET_CONFIGS = [
        ("longalpaca_pretokenized",               "long_alpaca"),
        ("maths_grade_school_pretokenized",        "maths_grade_school"),
        ("education_young_children_pretokenized",  "education_young_children"),
        ("megascience_pretokenized",               "megascience"),
        ("medquad_pretokenized",                   "medquad"),
        ("glaive_code_assistant_pretokenized",     "glaive_code_assistant"),
        ("glaive_function_calling_pretokenized",   "glaive_function_calling"),
        ("opus_3300_pretokenized",                 "opus_3300"),
        ("opus_10k_pretokenized",                  "opus_10k"),
    ]

    streaming_datasets = []
    for config_name, key in HUB_DATASET_CONFIGS:
        cap = MAX_EXAMPLES_PER_DATASET.get(key)
        print(f"\nStreaming {hf_repo} / {config_name} (cap={cap})...")
        try:
            ds = load_dataset(
                hf_repo,
                name=config_name,
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not load {config_name}: {e}")
            print(f"  [WARN] Skipping {config_name} — upload it to the Hub to include it.")
            continue

        # Apply cap by taking first N examples (shuffle not supported in streaming)
        if cap is not None:
            print(f"  Capping {key} to {cap} examples")
            ds = ds.take(cap)

        streaming_datasets.append(ds)
        print(f"  Streaming {config_name}")

    if not streaming_datasets:
        raise RuntimeError(
            f"No datasets could be loaded from {hf_repo}. "
            "Check that the repo is accessible and at least one config is uploaded."
        )

    # Interleave streaming datasets
    if len(streaming_datasets) > 1:
        interleaved = interleave_datasets(
            streaming_datasets,
            probabilities=[1.0 / len(streaming_datasets)] * len(streaming_datasets),
            stopping_strategy="all_exhausted",
        )
    else:
        interleaved = streaming_datasets[0]

    print(f"\nStreaming interleaved dataset ready")
    return interleaved

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
        dtype=torch.bfloat16,  # Changed from float16 to bfloat16 for FLA kernel compatibility
        load_in_4bit=False,    # Disabled 4-bit to prevent massive dequantization bottlenecks on RTX 5090
    )

    # Phase 1: Execute all architectural replacements upfront
    for layer_idx, _ in REPLACEMENT_SCHEDULE:
        if is_main_process:
            print(f"Applying hybrid surgery to layer {layer_idx}...")
        model = apply_hybrid_surgery(model, layer_idx)

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
            "self_attn",
            "input_layernorm",
            "post_attention_layernorm",
        ],  # Full Rank for new components
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
    if args.use_pretokenized:
        if is_main_process:
            print("Loading pretokenized Stage 2 datasets from local disk...")
        stage2_dataset = load_stage2_pretokenized_dataset(args.pretokenized_dir)
        dataset_iter = iter(stage2_dataset)
        is_streaming = False
        current_idx = 0
    elif args.hub:
        if is_main_process:
            print(f"Streaming pretokenized Stage 2 datasets from Hub: {args.hub_repo}")
        stage2_dataset = load_stage2_from_hub(args.hub_repo)
        dataset_iter = iter(stage2_dataset)
        is_streaming = True
        current_idx = 0
    elif args.download:
        if is_main_process:
            print("Downloading Stage 2 datasets from HuggingFace Hub (raw, will pretokenize inline)...")
        stage2_dataset = load_stage2_downloaded_dataset()
        # Validate dataset has required columns
        assert "input_ids" in stage2_dataset.column_names, "Downloaded dataset must be pretokenized with input_ids column"
        assert "attention_mask" in stage2_dataset.column_names, "Downloaded dataset must be pretokenized with attention_mask column"
        dataset_iter = iter(stage2_dataset)
        is_streaming = False
        current_idx = 0
    else:
        raise ValueError("Must specify one of: --use_pretokenized, --hub, or --download")

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

        # Validate dataset columns (only once per phase)
        assert "input_ids" in buffered_dataset.column_names, f"Missing input_ids column in buffered dataset for layer {layer_idx}"
        assert "attention_mask" in buffered_dataset.column_names, f"Missing attention_mask column in buffered dataset for layer {layer_idx}"

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
            fp16=False,  # Disable fp16 - use bf16 for FLA kernel compatibility
            bf16=True,   # Enable bf16 - required for stable Triton kernels
            logging_steps=10,
            remove_unused_columns=True,
            save_strategy="no",  # We save manually after each phase
            max_seq_length=8192,
            packing=True,
            # DDP settings
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=buffered_dataset,
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