"""
Pretokenization Pipeline for Stage 2 Training Data

This script processes all Stage 2 datasets (LongAlpaca, LLaVA) into a unified
pretokenized format compatible with Gemma 4's chat template and the hybrid
GDN training pipeline.

Key Requirements:
- Gemma 4 uses <start_of_turn>user/model<end_of_turn> format
- Padding side: LEFT (as per Gemma 4 docs)
- Attention mask: 1 for real tokens, 0 for padding
- Must handle variable-length sequences with proper padding
- Output format: Arrow/Parquet for efficient streaming during training
"""

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, interleave_datasets
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import shutil
import subprocess

# ==============================================================================
# Dataset Mapping Functions (from datarules.md)
# ==============================================================================

def map_alpaca(row):
    """Maps Yukang/LongAlpaca-12k format to Gemma 4 chat template."""
    prompt = row.get("instruction", "").strip()
    response = row.get("output", "").strip()
    if not prompt or not response:
        return None  # Skip empty rows
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_maths_grade_school(row):
    """Maps ajibawa-2023/Maths-Grade-School format."""
    prompt = row.get("instruction", "").strip()
    response = row.get("output", "").strip()
    if not prompt or not response:
        return None
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_education_young_children(row):
    """Maps ajibawa-2023/Education-Young-Children format."""
    prompt = row.get("prompt", "").strip()
    response = row.get("text", "").strip()
    if not prompt or not response:
        return None
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_megascience(row):
    """Maps MegaScience/MegaScience format."""
    prompt = row.get("question", "").strip()
    response = row.get("answer", "").strip()
    if not prompt or not response:
        return None
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_medquad(row):
    """Maps keivalya/MedQuad-MedicalQnADataset format."""
    prompt = row.get("Question", "").strip()
    response = row.get("Answer", "").strip()
    if not prompt or not response:
        return None
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_glaive_code_assistant(row):
    """Maps glaiveai/glaive-code-assistant-v3 format."""
    prompt = row.get("question", "").strip()
    response = row.get("answer", "").strip()
    if not prompt or not response:
        return None
    return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

def map_glaive_function_calling(row):
    """Maps glaiveai/glaive-function-calling-v2 format (multiturn + agentic)."""
    system = row.get("system", "").strip()
    chat = row.get("chat", "").strip()

    if not system or not chat:
        return None

    # Combine system prompt and chat into Gemma 4 format
    full_text = f"<start_of_turn>system\n{system}<end_of_turn>\n{chat}"

    # Convert USER/ASSISTANT markers to Gemma 4 format
    full_text = full_text.replace("USER:", "<start_of_turn>user\n")
    full_text = full_text.replace("ASSISTANT:", "<start_of_turn>model\n")
    full_text = full_text.replace("FUNCTION RESPONSE:", "<start_of_turn>user\nFunction returned:\n")
    full_text = full_text.replace("<|endoftext|>", "<end_of_turn>")

    return {"text": full_text}

def map_opus_10k(row):
    """Maps Roman1111111/claude-opus-4.6-10000x format (CoT with reasoning)."""
    messages = row.get("messages", [])

    if not messages:
        return None

    # Build conversation from messages list
    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning", "").strip()

        if role == "system" and content:
            conversation.append(f"<start_of_turn>system\n{content}<end_of_turn>")
        elif role == "user" and content:
            conversation.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            # Include reasoning trace if present (CoT)
            if reasoning and content:
                full_response = f"{reasoning}\n\n{content}"
            elif content:
                full_response = content
            else:
                continue
            conversation.append(f"<start_of_turn>model\n{full_response}<end_of_turn>")

    if not conversation:
        return None

    return {"text": "\n".join(conversation)}

def map_opus_3300(row):
    """Maps Crownelius/Opus-4.6-Reasoning-3300x format (CoT with thinking)."""
    problem = row.get("problem", "").strip()
    thinking = row.get("thinking", "").strip()
    solution = row.get("solution", "").strip()

    if not problem or not solution:
        return None

    # Combine thinking (CoT) and solution
    full_response = f"{thinking}\n\n{solution}" if thinking else solution

    return {"text": f"<start_of_turn>user\n{problem}<end_of_turn>\n<start_of_turn>model\n{full_response}<end_of_turn>"}

# ==============================================================================
# PHASE 4 — Multimodal Alignment (DEFERRED)
# ==============================================================================
# The following map functions are preserved for Phase 4 multimodal alignment
# training. They must NOT be used during progressive warm-start (Phase 3) because:
#   1. GDN cross-attention is zero-initialized and dormant during warm-start.
#   2. multimodal_states are never injected into the forward pass in this phase.
#   3. <image> tags are stripped here, producing pure text — these patterns
#      dilute the instruct/CoT reasoning distribution without adding visual grounding.
#
# To activate, Phase 4 requires:
#   - A modified forward pass that injects vision_latents -> multimodal_states
#   - Actual SigLIP-encoded vision patches (not just stripped text)
#   - Cross-attention o_proj re-initialized from zero to allow gradient flow
# ==============================================================================

# def map_llava(row):
#     """Maps liuhaotian/LLaVA-Instruct-150K format to Gemma 4 chat template."""
#     convs = row.get("conversations", [])
#
#     if len(convs) < 2:
#         return None
#
#     prompt = convs[0].get("value", "").strip() if len(convs) > 0 else ""
#     response = convs[1].get("value", "").strip() if len(convs) > 1 else ""
#
#     if not prompt or not response:
#         return None
#
#     # Strip <image> tags for text-only structural warmup
#     prompt = prompt.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
#     if not prompt:
#         return None
#
#     return {"text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}

# def map_sharegpt4v(row):
#     """Maps Lin-Chen/ShareGPT4V format."""
#     convs = row.get("conversations", [])
#
#     if not convs:
#         return None
#
#     # Build multi-turn conversation
#     conversation = []
#     for conv in convs:
#         from_role = conv.get("from", "")
#         value = conv.get("value", "").strip()
#
#         if not value:
#             continue
#
#         # Strip <image> tags for text-only warmup
#         value = value.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
#
#         if not value:
#             continue
#
#         if from_role == "human":
#             conversation.append(f"<start_of_turn>user\n{value}<end_of_turn>")
#         elif from_role == "gpt":
#             conversation.append(f"<start_of_turn>model\n{value}<end_of_turn>")
#
#     if not conversation:
#         return None
#
#     return {"text": "\n".join(conversation)}

# def map_visual_cot(row):
#     """Maps deepcs233/Visual-CoT format (visual reasoning with CoT)."""
#     question = row.get("question", "").strip()
#     thought = row.get("thought", "").strip()
#     full_answer = row.get("full_answer", "").strip()
#
#     if not question or not full_answer:
#         return None
#
#     # Combine thought (CoT) and answer
#     response = f"{thought}\n\n{full_answer}" if thought else full_answer
#
#     return {"text": f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"}


# ==============================================================================
# Tokenization with Proper Padding
# ==============================================================================

def tokenize_batch(examples, tokenizer, max_length=8192):
    """
    Tokenizes a batch of examples with LEFT padding (Gemma 4 requirement).

    Args:
        examples: Dict with 'text' key containing formatted chat strings
        tokenizer: Gemma 4 tokenizer with padding_side='left'
        max_length: Maximum sequence length (default 8192 for Gemma 4)

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors
    """
    # Tokenize with left padding
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",  # Pad to max_length for uniform batches
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # Return lists for datasets library
        add_special_tokens=True,  # Gemma 4 needs BOS token
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

# ==============================================================================
# Dataset Loading & Processing
# ==============================================================================

def load_and_pretokenize_dataset(
    dataset_name,
    map_func,
    tokenizer,
    max_length=8192,
    streaming=False,
    num_samples=None,
    output_path=None,
    subset=None,
    sample_rate=None,
):
    """
    Loads a dataset, applies formatting, tokenizes, and saves to disk.

    Args:
        dataset_name: HuggingFace dataset identifier
        map_func: Function to convert raw rows to Gemma 4 chat format
        tokenizer: Gemma 4 tokenizer
        max_length: Max sequence length
        streaming: Whether to use streaming mode
        num_samples: Limit number of samples (for testing)
        output_path: Where to save pretokenized data
        subset: Dataset subset/configuration name (optional)
        sample_rate: Float between 0.0 and 1.0 to sample a percentage of the dataset
    """
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}" + (f" (subset: {subset})" if subset else ""))
    if sample_rate:
        print(f"Sample rate: {sample_rate * 100}%")
    print(f"{'='*80}")

    # Load dataset
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=streaming)

    # Apply sample rate if specified (before formatting to save processing time)
    if sample_rate is not None:
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {sample_rate}")

        print(f"Sampling {sample_rate * 100}% of dataset...")
        if streaming:
            # For streaming datasets, we need to filter based on index
            import random
            random.seed(42)  # Reproducible sampling
            ds = ds.filter(lambda x, idx: random.random() < sample_rate, with_indices=True)
        else:
            # For non-streaming, calculate exact number of samples
            total_size = len(ds)
            sample_size = int(total_size * sample_rate)
            print(f"Selecting {sample_size} out of {total_size} examples")
            indices = list(range(total_size))
            import random
            random.seed(42)
            random.shuffle(indices)
            ds = ds.select(indices[:sample_size])

    # Apply chat template formatting
    print("Applying chat template formatting...")
    ds = ds.map(map_func, remove_columns=ds.column_names if not streaming else None)

    # Filter out None values (empty/invalid rows)
    # HuggingFace datasets.map() converts None returns to rows with None values
    # We need to filter these out before tokenization
    print("Filtering out empty rows...")
    def is_valid_row(example):
        # Check if example is None or if 'text' field is missing/empty
        if example is None:
            return False
        text = example.get("text")
        if text is None:
            return False
        if not isinstance(text, str):
            return False
        if text.strip() == "":
            return False
        return True

    ds = ds.filter(is_valid_row)

    # Limit samples if specified (for testing)
    if num_samples is not None:
        if streaming:
            ds = ds.take(num_samples)
        else:
            ds = ds.select(range(min(num_samples, len(ds))))

    # Tokenize
    print("Tokenizing...")
    ds_tokenized = ds.map(
        lambda examples: tokenize_batch(examples, tokenizer, max_length),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
    )

    # Final validation: Filter out any rows with all-zero input_ids
    print("Final validation: removing any all-zero sequences...")
    def has_valid_tokens(example):
        # Check if input_ids has at least one non-zero token
        input_ids = example.get("input_ids", [])
        if not input_ids:
            return False
        # Check if there's at least one non-padding token
        attention_mask = example.get("attention_mask", [])
        if not attention_mask:
            return False
        return sum(attention_mask) > 0

    ds_tokenized = ds_tokenized.filter(has_valid_tokens)
    print(f"After validation: {len(ds_tokenized)} valid examples")

    # Save to disk if output path specified
    if output_path:
        print(f"Saving to {output_path}...")
        if streaming:
            # For streaming datasets, we need to materialize first
            materialized = []
            for item in tqdm(ds_tokenized, desc="Materializing"):
                materialized.append(item)
            ds_tokenized = Dataset.from_list(materialized)

        ds_tokenized.save_to_disk(output_path)
        print(f"Saved {len(ds_tokenized)} examples")

    return ds_tokenized

# ==============================================================================
# HuggingFace Upload & Cleanup Functions
# ==============================================================================

def check_dataset_exists_on_hf(hf_repo, dataset_name):
    """
    Check if a dataset already exists in the HuggingFace repository.

    Args:
        hf_repo: HuggingFace repository name (e.g., "leonidas123/gemma-4-pretokenized-traces")
        dataset_name: Name of the dataset (e.g., "longalpaca_pretokenized")

    Returns:
        bool: True if dataset exists, False otherwise
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # List all files in the repo
        files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")

        # Check both possible locations:
        # 1. pretokenized_data/{dataset_name}/ (new structure)
        # 2. {dataset_name}/ (legacy structure)
        dataset_prefix_new = f"pretokenized_data/{dataset_name}/"
        dataset_prefix_legacy = f"{dataset_name}/"

        exists = any(f.startswith(dataset_prefix_new) or f.startswith(dataset_prefix_legacy) for f in files)

        if exists:
            print(f"✓ Dataset {dataset_name} already exists in {hf_repo}")
        else:
            print(f"✗ Dataset {dataset_name} not found in {hf_repo}")

        return exists

    except Exception as e:
        print(f"⚠ Could not check if {dataset_name} exists: {e}")
        # If we can't check, assume it doesn't exist to avoid skipping
        return False

def upload_to_huggingface(dataset_path, hf_repo, dataset_name):
    """
    Upload a pretokenized dataset to HuggingFace Hub.

    Args:
        dataset_path: Local path to the dataset directory
        hf_repo: HuggingFace repository name (e.g., "leonidas123/gemma-4-pretokenized-traces")
        dataset_name: Name of the dataset (e.g., "longalpaca_pretokenized")
    """
    print(f"\n{'='*80}")
    print(f"Uploading {dataset_name} to {hf_repo}")
    print(f"{'='*80}")

    try:
        # Load the dataset
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)

        # Upload to HuggingFace Hub with path structure: pretokenized_data/{dataset_name}
        repo_path = f"pretokenized_data/{dataset_name}"

        print(f"Pushing to hub: {hf_repo}/{repo_path}")
        ds.push_to_hub(
            hf_repo,
            config_name=dataset_name,
            split="train",
            private=False,
        )

        print(f"✓ Successfully uploaded {dataset_name}")
        return True

    except Exception as e:
        print(f"✗ Failed to upload {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_dataset(dataset_path, dataset_name):
    """
    Clean up local dataset files and HuggingFace cache.

    Args:
        dataset_path: Local path to the dataset directory
        dataset_name: Name of the dataset for cache identification
    """
    print(f"\n{'='*80}")
    print(f"Cleaning up {dataset_name}")
    print(f"{'='*80}")

    try:
        # Delete local dataset directory
        if Path(dataset_path).exists():
            print(f"Deleting local dataset: {dataset_path}")
            shutil.rmtree(dataset_path)
            print(f"✓ Deleted {dataset_path}")

        # Clear HuggingFace cache for this dataset
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        if cache_dir.exists():
            print(f"Clearing HuggingFace cache...")
            # Find and delete cache entries for this dataset
            for cache_entry in cache_dir.glob("*"):
                if dataset_name.replace("_pretokenized", "") in cache_entry.name.lower():
                    try:
                        if cache_entry.is_dir():
                            shutil.rmtree(cache_entry)
                        else:
                            cache_entry.unlink()
                        print(f"✓ Deleted cache: {cache_entry.name}")
                    except Exception as e:
                        print(f"⚠ Could not delete cache {cache_entry.name}: {e}")

        print(f"✓ Cleanup complete for {dataset_name}")
        return True

    except Exception as e:
        print(f"✗ Cleanup failed for {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def upload_metadata_to_hf(metadata_path, hf_repo):
    """
    Upload metadata.json to HuggingFace Hub.

    Args:
        metadata_path: Local path to metadata.json
        hf_repo: HuggingFace repository name
    """
    print(f"\n{'='*80}")
    print(f"Uploading metadata.json to {hf_repo}")
    print(f"{'='*80}")

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="pretokenized_data/metadata.json",
            repo_id=hf_repo,
            repo_type="dataset",
        )

        print(f"✓ Successfully uploaded metadata.json")
        return True

    except Exception as e:
        print(f"✗ Failed to upload metadata.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Pretokenize Stage 2 training datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pretokenized_data",
        help="Directory to save pretokenized datasets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples per dataset (for testing)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Gemma 4 model name for tokenizer"
    )
    parser.add_argument(
        "--upload_to_hf",
        action="store_true",
        help="Upload pretokenized datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="leonidas123/gemma-4-pretokenized-traces",
        help="HuggingFace repository to upload to"
    )
    parser.add_argument(
        "--cleanup_after_upload",
        action="store_true",
        help="Delete local files and caches after successful upload"
    )
    parser.add_argument(
        "--dataset_sample_rate",
        type=str,
        default=None,
        help="Per-dataset sample rates as JSON dict (e.g., '{\"glaive_function_calling_pretokenized\": 0.1}') or single float for all datasets"
    )

    args = parser.parse_args()

    # Parse dataset sample rates
    dataset_sample_rates = {}
    if args.dataset_sample_rate:
        try:
            # Try parsing as JSON dict first
            dataset_sample_rates = json.loads(args.dataset_sample_rate)
            if not isinstance(dataset_sample_rates, dict):
                raise ValueError("Must be a dict")
        except (json.JSONDecodeError, ValueError):
            # Try parsing as single float for all datasets
            try:
                global_rate = float(args.dataset_sample_rate)
                if not 0.0 < global_rate <= 1.0:
                    raise ValueError("Sample rate must be between 0.0 and 1.0")
                print(f"Using global sample rate: {global_rate * 100}% for all datasets")
            except ValueError as e:
                print(f"ERROR: Invalid --dataset_sample_rate format: {e}")
                print("Expected: JSON dict like '{\"glaive_function_calling_pretokenized\": 0.1}' or single float like '0.1'")
                return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer with LEFT padding (Gemma 4 requirement)
    print("Loading Gemma 4 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",  # CRITICAL: Gemma 4 uses left padding
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer config:")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Padding side: {tokenizer.padding_side}")
    print(f"  - Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Dataset configurations
    datasets_config = [
        # Long-Context Domain
        {
            "name": "Yukang/LongAlpaca-12k",
            "map_func": map_alpaca,
            "output_name": "longalpaca_pretokenized",
        },
        # Instruction-Tuned Domain (non-CoT)
        {
            "name": "ajibawa-2023/Maths-Grade-School",
            "map_func": map_maths_grade_school,
            "output_name": "maths_grade_school_pretokenized",
        },
        {
            "name": "ajibawa-2023/Education-Young-Children",
            "map_func": map_education_young_children,
            "output_name": "education_young_children_pretokenized",
        },
        {
            "name": "MegaScience/MegaScience",
            "map_func": map_megascience,
            "output_name": "megascience_pretokenized",
        },
        {
            "name": "keivalya/MedQuad-MedicalQnADataset",
            "map_func": map_medquad,
            "output_name": "medquad_pretokenized",
        },
        {
            "name": "glaiveai/glaive-code-assistant-v3",
            "map_func": map_glaive_code_assistant,
            "output_name": "glaive_code_assistant_pretokenized",
        },
        # Multiturn + Agentic
        {
            "name": "glaiveai/glaive-function-calling-v2",
            "map_func": map_glaive_function_calling,
            "output_name": "glaive_function_calling_pretokenized",
        },
        # CoT Reasoning Datasets
        {
            "name": "Roman1111111/claude-opus-4.6-10000x",
            "map_func": map_opus_10k,
            "output_name": "opus_10k_pretokenized",
        },
        {
            "name": "Crownelius/Opus-4.6-Reasoning-3300x",
            "map_func": map_opus_3300,
            "output_name": "opus_3300_pretokenized",
        },
        # Vision-Language Domain — DEFERRED to Phase 4 (Multimodal Alignment)
        # Requires a modified forward pass injecting vision_latents -> multimodal_states
        # and active cross-attention. See commented map functions above.
        # {
        #     "name": "liuhaotian/LLaVA-Instruct-150K",
        #     "map_func": map_llava,
        #     "output_name": "llava_pretokenized",
        # },
        # {
        #     "name": "Lin-Chen/ShareGPT4V",
        #     "map_func": map_sharegpt4v,
        #     "output_name": "sharegpt4v_pretokenized",
        #     "subset": "ShareGPT4V-PT",
        # },
        # {
        #     "name": "deepcs233/Visual-CoT",
        #     "map_func": map_visual_cot,
        #     "output_name": "visual_cot_pretokenized",
        # },
    ]

    # Process each dataset
    pretokenized_paths = []
    for config in datasets_config:
        output_path = output_dir / config["output_name"]

        # Check if dataset already exists on HuggingFace (if uploading)
        if args.upload_to_hf:
            if check_dataset_exists_on_hf(args.hf_repo, config["output_name"]):
                print(f"⏭ Skipping {config['output_name']} - already exists on HuggingFace")
                continue

        # Determine sample rate for this dataset
        sample_rate = None
        if args.dataset_sample_rate:
            if isinstance(dataset_sample_rates, dict):
                sample_rate = dataset_sample_rates.get(config["output_name"])
                if sample_rate:
                    print(f"Using {sample_rate * 100}% sample rate for {config['output_name']}")
            else:
                sample_rate = global_rate

        try:
            load_and_pretokenize_dataset(
                dataset_name=config["name"],
                map_func=config["map_func"],
                tokenizer=tokenizer,
                max_length=args.max_length,
                streaming=False,  # Use non-streaming for pretokenization
                num_samples=args.num_samples,
                output_path=str(output_path),
                subset=config.get("subset", None),
                sample_rate=sample_rate,
            )
            pretokenized_paths.append(str(output_path))

            # Upload to HuggingFace if requested
            if args.upload_to_hf:
                upload_success = upload_to_huggingface(
                    dataset_path=str(output_path),
                    hf_repo=args.hf_repo,
                    dataset_name=config["output_name"]
                )

                # Cleanup after successful upload
                if upload_success and args.cleanup_after_upload:
                    cleanup_dataset(
                        dataset_path=str(output_path),
                        dataset_name=config["output_name"]
                    )

        except Exception as e:
            print(f"ERROR processing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "datasets": pretokenized_paths,
        "total_datasets": len(pretokenized_paths),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("Pretokenization Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets processed: {len(pretokenized_paths)}")
    print(f"Metadata saved to: {metadata_path}")

    # Upload metadata to HuggingFace if requested
    if args.upload_to_hf:
        upload_metadata_to_hf(metadata_path, args.hf_repo)

        # Cleanup metadata after upload
        if args.cleanup_after_upload:
            print(f"\nDeleting local metadata: {metadata_path}")
            metadata_path.unlink()
            print(f"✓ Deleted {metadata_path}")

    # Final summary
    if args.upload_to_hf:
        print(f"\n{'='*80}")
        print("Upload Summary")
        print(f"{'='*80}")
        print(f"Repository: {args.hf_repo}")
        print(f"Datasets uploaded: {len(pretokenized_paths)}")
        if args.cleanup_after_upload:
            print("Local files and caches: CLEANED")
        else:
            print("Local files: RETAINED")
        print(f"\nView at: https://huggingface.co/datasets/{args.hf_repo}")

if __name__ == "__main__":
    main()
