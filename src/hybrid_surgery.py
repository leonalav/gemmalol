import sys
import os
import types
import gc
from tqdm import tqdm
"""
Implement before surgery (for whatever LLMs are editing this, do NOT REMOVE THE LINES BELOW):
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=900
export HF_HUB_DISABLE_XET=1
"""

# Path fixing implementation for standalone execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import torch.nn as nn

from src.architecture.gdn_hybrid_layer import GemmaDeltaNetLayer
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRouter,
    Gemma4TextExperts,
    Gemma4TextDecoderLayer,
    Gemma4ModelOutputWithPast,
    BaseModelOutputWithPast
)

def perform_gdn_surgery(model):
    """
    Mutates a multimodal Gemma 4 model (e.g. Gemma4ForConditionalGeneration) 
    into a GDN hybrid in-place.
    """
    # 1. Target the text backbone
    # Potential structures:
    # A. Gemma4ForConditionalGeneration -> model (Gemma4Model) -> language_model (Gemma4TextModel)
    # B. Gemma4Model -> language_model (Gemma4TextModel)
    # C. Gemma4TextModel directly
    
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        # top-level "ForConditionalGeneration" model
        text_model = model.model.language_model
        multimodal_model = model.model
    elif hasattr(model, "language_model"):
        # top-level "Model" (multimodal)
        text_model = model.language_model
        multimodal_model = model
    elif hasattr(model, "layers"):
        # direct text backbone
        text_model = model
        multimodal_model = None
    else:
        # Fallback for generic transformer structures
        text_model = getattr(model, "model", model)
        multimodal_model = None

    if not hasattr(text_model, "layers"):
        raise AttributeError(f"Could not find 'layers' in target text model: {type(text_model)}. "
                             "Please verify the model structure.")

    print(f"Starting surgery on target text model: {type(text_model)}...")
    
    # --- Freeze Base Model ---
    model.requires_grad_(False)
    
    config = text_model.config
    
    # 2. Layer-by-layer replacement (Optimized for Memory)
    print(f"Surgically replacing {len(text_model.layers)} layers...")
    for i in tqdm(range(len(text_model.layers)), desc="Gemma->GDN Surgery"):
        # Access old layer and create new layer
        old_layer = text_model.layers[i]
        new_layer = GemmaDeltaNetLayer(config, layer_idx=i)
        
        # --- Weight Transfer (Decoupled LayerNorms) ---
        # [ARCH REPAIR]: Use deepcopy & unfreeze to allow norms to adapt to 
        # the new GDN/Cross-Attention feature distributions.
        import copy
        new_layer.input_layernorm = copy.deepcopy(old_layer.input_layernorm).requires_grad_(True)
        new_layer.post_attention_layernorm = copy.deepcopy(old_layer.post_attention_layernorm).requires_grad_(True)
        
        # FFN norms stay frozen/pointed (shared with MoE experts)
        new_layer.pre_feedforward_layernorm = old_layer.pre_feedforward_layernorm
        new_layer.post_feedforward_layernorm = old_layer.post_feedforward_layernorm
        
        # Inherit the massive FFN / MoE blocks directly
        new_layer.mlp = old_layer.mlp
        if hasattr(old_layer, "enable_moe_block") and old_layer.enable_moe_block:
            new_layer.router = old_layer.router
            new_layer.experts = old_layer.experts
            new_layer.post_feedforward_layernorm_1 = old_layer.post_feedforward_layernorm_1
            new_layer.post_feedforward_layernorm_2 = old_layer.post_feedforward_layernorm_2
            new_layer.pre_feedforward_layernorm_2 = old_layer.pre_feedforward_layernorm_2
        
        new_layer.layer_scalar = old_layer.layer_scalar
        
        if new_layer.hidden_size_per_layer_input > 0:
            new_layer.per_layer_input_gate = old_layer.per_layer_input_gate
            new_layer.per_layer_projection = old_layer.per_layer_projection
            new_layer.post_per_layer_input_norm = old_layer.post_per_layer_input_norm

        # --- Weight Transfer for QKV & O Projections ---
        # Initialize GDN projections with pretrained Gemma weights (warm start)
        new_layer.self_attn.q_proj.weight.data.copy_(old_layer.self_attn.q_proj.weight.data)
        new_layer.self_attn.k_proj.weight.data.copy_(old_layer.self_attn.k_proj.weight.data)
        new_layer.self_attn.v_proj.weight.data.copy_(old_layer.self_attn.v_proj.weight.data)
        new_layer.self_attn.o_proj.weight.data.copy_(old_layer.self_attn.o_proj.weight.data)
        
        # --- Unfreeze New Components ---
        new_layer.self_attn.requires_grad_(True)
        
        # --- In-Place Substitution & Memory Cleanup ---
        text_model.layers[i] = new_layer
        del old_layer
        if i % 5 == 0: # Periodic aggressive cleanup to avoid slowing down too much
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 3. Forward Pass Patching (Tunneling & Suppression) ---
    
    # A. Patch Gemma4TextModel to support PLE correctly without cross attention
    def patched_text_model_forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # We simplify the forward slightly for the patch, but preserve core logic
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        per_layer_inputs = kwargs.get("per_layer_inputs")

        if self.hidden_size_per_layer_input and per_layer_inputs is None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        # Iterate through hybrid layers
        next_decoder_cache = [] if kwargs.get("use_cache") else None

        for i, layer in enumerate(self.layers):
            current_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            # Index past_key_values for this layer if provided
            layer_past = None
            if kwargs.get("past_key_values") is not None:
                layer_past = kwargs["past_key_values"][i]

            # Check if this is a hybrid layer (GemmaDeltaNetLayer) or original layer
            layer_output = layer(
                hidden_states,
                per_layer_input=current_ple,
                past_key_values=layer_past,
                **kwargs
            )

            # Handle both tuple returns (hybrid layers) and BaseModelOutputWithPast (original layers)
            if isinstance(layer_output, tuple):
                hidden_states, layer_state = layer_output
            else:
                hidden_states = layer_output.last_hidden_state if hasattr(layer_output, 'last_hidden_state') else layer_output
                layer_state = layer_output.past_key_values if hasattr(layer_output, 'past_key_values') else None

            if next_decoder_cache is not None:
                next_decoder_cache.append(layer_state)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
        )

    text_model.forward = types.MethodType(patched_text_model_forward, text_model)

    print("Surgery complete. Early-fusion active.")
    return model

def save_hybrid_delta_checkpoint(model, save_path="gdn_delta_weights.pt"):
    """Saves ONLY the trainable GDN and Cross-Attention layers."""
    print("Extracting trainable Delta weights...")
    
    # Extract only parameters that require gradients
    delta_state_dict = {
        name: param.cpu() 
        for name, param in model.named_parameters() 
        if param.requires_grad
    }
    
    torch.save(delta_state_dict, save_path)
    
    print(f"✅ Saved delta checkpoint to {save_path}.")
    print(f"Total tensors saved: {len(delta_state_dict)}")

def load_hybrid_delta_checkpoint(model, checkpoint_path):
    """Loads delta weights into a surgically modified model."""
    print(f"Loading delta weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Use strict=False because state_dict only contains the delta
    model.load_state_dict(state_dict, strict=False)
    print("✅ Delta weights loaded.")

def verify_architecture(model):
    """
    Prints the layer structure of the mutated model for manual verification.
    """
    # Find the text backbone again
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    elif hasattr(model, "language_model"):
        text_model = model.language_model
    elif hasattr(model, "layers"):
        text_model = model
    else:
        text_model = getattr(model, "model", model)

    print("\n--- Mutated Architecture Verification ---")
    print(f"Top-level Model Class: {type(model)}")
    print(f"Text Backbone Class: {type(text_model)}")
    print(f"Total Layers: {len(text_model.layers)}")
    
    for i, layer in enumerate(text_model.layers):
        print(f"Layer {i:02d}: {type(layer).__name__}")
        # Verify recurrent component
        if hasattr(layer, "self_attn") and type(layer.self_attn).__name__ == "GatedDeltaNet":
            print(f"  - Recurrent Op: GatedDeltaNet")
    print("------------------------------------------\n")

def setup_unsloth_training(model):
    """
    Applies Unsloth PEFT/LoRA patching to the hybrid model.
    """
    from unsloth import FastLanguageModel
    
    # [PEFT REPAIR]: Target ONLY frozen pre-existing layers with LoRA.
    # New architecture components (GDN, Cross-Attn) must be in modules_to_save
    # to train in FULL RANK, avoiding the random-noise-lock trap.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["gate_proj", "up_proj", "down_proj"], # LoRA on original MLP/Experts
        modules_to_save = [
            "self_attn",
            "input_layernorm",
            "post_attention_layernorm",
        ], # Full Rank for new components
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    return model

if __name__ == "__main__":
    # Structural check with configuration only (no weights needed for logic verification)
    from transformers import AutoConfig, AutoModel
    import traceback
    
    print("Testing surgery on dummy config...")
    try:
        # Attempt to load config for E2B
        config = AutoConfig.from_pretrained("google/gemma-4-E2B-it", trust_remote_code=True)
        # Instantiate dummy model for surgery test
        model = AutoModel.from_config(config, trust_remote_code=True) 
        
        model = perform_gdn_surgery(model)
        print("Structural verification successful! Surgery performed correctly.")
        
        # New: Print detailed architecture
        verify_architecture(model)
        
    except Exception as e:
        print(f"Structural verification failed: {e}")
        traceback.print_exc()
        
    # --- 4. Delta Checkpoint Export ---
    # We no longer use model.save_pretrained() by default as it saves 17GB+ of frozen weights.
    # Instead, we save only the newly initialized GDN and Cross-Attention delta.
    delta_path = "gdn_delta_weights.pt"
    save_hybrid_delta_checkpoint(model, delta_path)
