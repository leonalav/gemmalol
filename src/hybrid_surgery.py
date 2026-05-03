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

def perform_gdn_surgery(model, target_layers=None):
    """
    Mutates a multimodal Gemma 4 model (e.g. Gemma4ForConditionalGeneration) 
    into a GDN hybrid in-place. If target_layers is provided, only those layers 
    will be mutated.
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
    layers_to_replace = target_layers if target_layers is not None else range(len(text_model.layers))
    print(f"Surgically replacing {len(layers_to_replace)} layers...")
    for i in tqdm(layers_to_replace, desc="Gemma->GDN Surgery"):
        # Access old layer and create new layer
        old_layer = text_model.layers[i]
        new_layer = GemmaDeltaNetLayer(config, layer_idx=i)
        
        # --- Weight Transfer (Decoupled LayerNorms) ---
        # [MEMORY REPAIR]: Replaced deepcopy with direct assignment to prevent 1.5GB spikes on T4
        new_layer.input_layernorm = old_layer.input_layernorm
        new_layer.post_attention_layernorm = old_layer.post_attention_layernorm
        
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
        # Handle heterogeneous Gemma 4 layouts (Fused QKV vs Split Q/K/V)
        old_attn = old_layer.self_attn
        new_attn = new_layer.self_attn

        if hasattr(old_attn, "qkv_proj"):
            # Sliding Window / Fused layout
            w_qkv = old_attn.qkv_proj.weight.data
            out_features = w_qkv.shape[0]
            
            # Dynamically calculate split based on model heads
            q_dim = new_attn.num_heads * new_attn.head_dim
            k_dim = new_attn.num_key_value_heads * new_attn.head_dim
            v_dim = k_dim
            
            # [SHAPE REPAIR]: Resize projections if they don't match the checkpoint
            if q_dim + k_dim + v_dim != out_features:
                print(f"Warning: Layer {i} QKV mismatch. Resizing new_attn to match {out_features} output features.")
                total_heads = new_attn.num_heads + 2 * new_attn.num_key_value_heads
                unit_dim = out_features // total_heads
                new_attn.head_dim = unit_dim
                # Re-init projections with correct shapes
                new_attn.q_proj = nn.Linear(new_attn.q_proj.in_features, new_attn.num_heads * unit_dim, bias=new_attn.q_proj.bias is not None).to(w_qkv.device, w_qkv.dtype)
                new_attn.k_proj = nn.Linear(new_attn.k_proj.in_features, new_attn.num_key_value_heads * unit_dim, bias=new_attn.k_proj.bias is not None).to(w_qkv.device, w_qkv.dtype)
                new_attn.v_proj = nn.Linear(new_attn.v_proj.in_features, new_attn.num_key_value_heads * unit_dim, bias=new_attn.v_proj.bias is not None).to(w_qkv.device, w_qkv.dtype)
                q_dim = new_attn.num_heads * unit_dim
                k_dim = v_dim = new_attn.num_key_value_heads * unit_dim

            w_q, w_k, w_v = w_qkv.split([q_dim, k_dim, v_dim], dim=0)
            new_attn.q_proj.weight.data.copy_(w_q)
            new_attn.k_proj.weight.data.copy_(w_k)
            new_attn.v_proj.weight.data.copy_(w_v)
        else:
            # Global / Split layout
            if hasattr(old_attn, "q_proj"):
                old_q = old_attn.q_proj.weight.data
                if new_attn.q_proj.weight.data.shape != old_q.shape:
                    print(f"Resizing new_attn.q_proj for layer {i} from {new_attn.q_proj.weight.data.shape} to {old_q.shape}")
                    new_attn.q_proj = nn.Linear(old_q.shape[1], old_q.shape[0], bias=new_attn.q_proj.bias is not None).to(old_q.device, old_q.dtype)
                new_attn.q_proj.weight.data.copy_(old_q)
            
            if hasattr(old_attn, "k_proj"):
                old_k = old_attn.k_proj.weight.data
                if new_attn.k_proj.weight.data.shape != old_k.shape:
                    new_attn.k_proj = nn.Linear(old_k.shape[1], old_k.shape[0], bias=new_attn.k_proj.bias is not None).to(old_k.device, old_k.dtype)
                new_attn.k_proj.weight.data.copy_(old_k)
                
            if hasattr(old_attn, "v_proj"):
                old_v = old_attn.v_proj.weight.data
                if new_attn.v_proj.weight.data.shape != old_v.shape:
                    new_attn.v_proj = nn.Linear(old_v.shape[1], old_v.shape[0], bias=new_attn.v_proj.bias is not None).to(old_v.device, old_v.dtype)
                new_attn.v_proj.weight.data.copy_(old_v)

        # [SHAPE REPAIR]: Resize o_proj if input dimension doesn't match
        old_o = old_attn.o_proj.weight.data
        if new_attn.o_proj.weight.data.shape[1] != old_o.shape[1]:
            print(f"Resizing new_attn.o_proj for layer {i} from {new_attn.o_proj.weight.data.shape} to {old_o.shape}")
            new_attn.o_proj = nn.Linear(old_o.shape[1], old_o.shape[0], bias=new_attn.o_proj.bias is not None).to(old_o.device, old_o.dtype)
            
        new_attn.o_proj.weight.data.copy_(old_o)
        
        # --- Unfreeze New Components ---
        new_layer.self_attn.requires_grad_(True)
        
        # --- In-Place Substitution & Memory Cleanup ---
        text_model.layers[i] = new_layer
        del old_layer
        # Aggressive cleanup for T4 DDP
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

        # [REPAIR]: Compute RoPE embeddings (required for Gemma 4 layers)
        # position_ids is passed from the top-level forward
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if hasattr(self, "rotary_emb") else None

        for i, layer in enumerate(self.layers):
            current_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None #idk

            # Index past_key_values for this layer if provided
            layer_past = None
            if kwargs.get("past_key_values") is not None:
                layer_past = kwargs["past_key_values"][i]

            # Check if this is a hybrid layer (GemmaDeltaNetLayer) or original layer
            # [REPAIR]: Prevent "multiple values for keyword argument" by cleaning kwargs
            layer_kwargs = kwargs.copy()
            layer_kwargs.pop("past_key_values", None)
            layer_kwargs.pop("use_cache", None)

            layer_output = layer(
                hidden_states,
                per_layer_input=current_ple,
                past_key_values=layer_past,
                position_embeddings=position_embeddings,
                **layer_kwargs
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
