"""
Gated DeltaNet Hybrid Layer — surgical replacement for Gemma4TextDecoderLayer.

Key API contracts this module must satisfy:

1. **FLA chunk_gated_delta_rule** (fla.ops.gated_delta_rule)
   - Tensor layout: [B, T, H, K]  (NOT [B, H, T, K])
   - All tensors (q, k, v, g, beta) must share the SAME dtype
     because `h = k.new_empty(...)` inherits k's dtype, and
     Triton `tl.dot(b_q, b_h)` requires matching dtypes.
   - g (gate): raw input when use_gate_in_kernel=True. The kernel
     internally fuses `-exp(A_log) * softplus(g + dt_bias)`.
   - beta: [B, T, H] — sigmoid-activated update strength.

2. **Gemma4TextDecoderLayer** (transformers.models.gemma4)
   - forward() returns a BARE torch.Tensor (not a tuple).
   - The caller in Gemma4TextModel does `hidden_states = decoder_layer(...)`.
   - Signature: forward(hidden_states, per_layer_input, ..., **kwargs).

3. **Unsloth gradient checkpointing**
   - Compatible with torch.compile / dynamo. Avoid .float() on
     non-tensor objects (tuples) — this crashes torchdynamo.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) if 'src' in current_dir else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Disable Triton autotuning noise
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRouter,
    Gemma4TextExperts,
    Gemma4TextDecoderLayer,
)


# ============================================================================
# GatedDeltaNet — self-attention replacement (Section 3.4 of the GDN paper)
# ============================================================================

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet block following Section 3.4 "Block Design".

    Data flow:
        x → Linear → ShortConv(silu) → [q: L2-norm] → chunk_gated_delta_rule → FusedRMSNormGated → o_proj → out
                                                 ↗ Linear → sigmoid (β)
                                          x → Linear → raw (g, gate)
    """

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int = None, head_dim: int = 128,
                 conv_size: int = 4, layer_idx: int = -1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim
        self.kv_dim = self.num_key_value_heads * head_dim
        self.layer_idx = layer_idx

        # QKV path: Linear → ShortConvolution(silu)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)

        self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation='silu')
        self.k_conv1d = ShortConvolution(hidden_size=self.kv_dim, kernel_size=conv_size, activation='silu')
        self.v_conv1d = ShortConvolution(hidden_size=self.kv_dim, kernel_size=conv_size, activation='silu')

        # Gate (g) and beta paths: Linear only
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # Mamba2-style parameters for the fused gate kernel
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(0.001, 16)))
        self.dt_bias = nn.Parameter(torch.randn(num_heads))

        # Output gate + projection
        self.g_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(self.key_dim, hidden_size, bias=False)

    def _unpack_tuple(self, x):
        """LoRA-wrapped projections may return (output, extras). Extract just the tensor."""
        return x[0] if isinstance(x, tuple) else x

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.shape

        # --- QKV: Linear → Conv1d(+SiLU) ---
        q = self._unpack_tuple(self.q_conv1d(self._unpack_tuple(self.q_proj(x))))
        k = self._unpack_tuple(self.k_conv1d(self._unpack_tuple(self.k_proj(x))))
        v = self._unpack_tuple(self.v_conv1d(self._unpack_tuple(self.v_proj(x))))

        # L2-norm on q (paper fidelity)
        q = F.normalize(q, p=2, dim=-1)

        # Reshape to FLA's expected layout: [B, T, H, D]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.num_key_value_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.num_key_value_heads)

        # Handle GQA (repeat K/V heads to match Q heads)
        if self.num_key_value_heads != self.num_heads:
            repeats = self.num_heads // self.num_key_value_heads
            k = torch.repeat_interleave(k, repeats, dim=2)
            v = torch.repeat_interleave(v, repeats, dim=2)

        # --- Gate (g) and Beta ---
        # g: raw input to use_gate_in_kernel=True. The FLA kernel internally
        # computes: -exp(A_log) * softplus(g + dt_bias), then chunk-cumsum.
        g = self._unpack_tuple(self.a_proj(x))          # [B, T, H]

        beta = self._unpack_tuple(self.b_proj(x))
        beta = beta.sigmoid()                            # [B, T, H]

        # --- Dtype unification ---
        # FLA allocates h = k.new_empty(...), so h inherits k's dtype.
        # Triton tl.dot(b_q, b_h) requires matching dtypes.
        # All tensors must be the same dtype — use bf16 (model dtype).
        target_dtype = torch.bfloat16
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
        g, beta  = g.to(target_dtype), beta.to(target_dtype)

        # Cast parameters at use-time (no in-place .data mutation)
        A_log  = self.A_log.to(target_dtype)
        dt_bias = self.dt_bias.to(target_dtype)

        # Ensure contiguity for Triton
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        g, beta = g.contiguous(), beta.contiguous()

        # --- FLA Kernel ---
        o, final_state = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            g=g,
            beta=beta,
            initial_state=state,
            output_final_state=not self.training,
            use_qk_l2norm_in_kernel=False,   # we did L2-norm manually
            use_gate_in_kernel=True,          # fuse -exp(A_log)*softplus(g+dt_bias) inside
            A_log=A_log,
            dt_bias=dt_bias,
        )

        # o: [B, T, H, D] → reshape for gating
        o = o.to(x.dtype)   # cast back to model dtype for downstream ops
        o = rearrange(o, 'b t h d -> b t h d')  # already [B,T,H,D], identity reshape

        # --- Output gate: g_proj(x) → SiLU (inside FusedRMSNormGated) ---
        g_gate = self._unpack_tuple(self.g_proj(x))
        g_gate = rearrange(g_gate, 'b t (h d) -> b t h d', h=self.num_heads)

        o = self.o_norm(o, g_gate)
        o = rearrange(o, 'b t h d -> b t (h d)')

        out = self._unpack_tuple(self.o_proj(o))
        return out, final_state


# ============================================================================
# GemmaDeltaNetLayer — drop-in replacement for Gemma4TextDecoderLayer
# ============================================================================

class GemmaDeltaNetLayer(nn.Module):
    """
    Surgical replacement for Gemma4TextDecoderLayer.

    CRITICAL: forward() must return a BARE torch.Tensor, matching the
    Gemma4TextModel contract: `hidden_states = decoder_layer(...)`.
    Returning a tuple would make the next layer's input_layernorm
    receive a tuple instead of a tensor → crash.
    """

    def __init__(self, config, layer_idx: int, use_cross_attn: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)

        # Norms (matching Gemma4 exactly)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # GatedDeltaNet (replaces self_attn)
        self.self_attn = GatedDeltaNet(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=getattr(config, "head_dim", 128),
            layer_idx=layer_idx,
        )

        # Standard Gemma4 FFN / MoE
        self.mlp = Gemma4TextMLP(config, layer_idx)

        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.register_buffer("layer_scalar", torch.ones(1))

        # PLE (Per-Layer Embedding) support
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input > 0:
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        multimodal_states: Optional[torch.Tensor] = None,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass matching Gemma4TextDecoderLayer signature.
        Returns a BARE tensor (not a tuple) — this is critical.
        """
        residual = hidden_states

        # --- PRE-NORM ---
        hidden_states = self.input_layernorm(hidden_states)

        # --- 1. GATED DELTANET ---
        # Sanitize past_key_values: HF DynamicCache objects crash Triton
        gdn_state = past_key_values
        if gdn_state is not None and not isinstance(gdn_state, torch.Tensor):
            if isinstance(gdn_state, tuple) and len(gdn_state) > 0 and isinstance(gdn_state[0], torch.Tensor):
                gdn_state = gdn_state[0]
            else:
                gdn_state = None

        gdn_out, _ = self.self_attn(hidden_states, state=gdn_state)

        hidden_states = gdn_out

        # --- POST-ATTENTION NORM + RESIDUAL ---
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- 3. FFN / MoE ---
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            # Matches original Gemma4 MoE ordering exactly (see modeling_gemma4.py L1394-1406)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- 4. PLE (Per-Layer Embedding) ---
        if self.hidden_size_per_layer_input > 0:
            ple_residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            if per_layer_input is not None:
                hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = ple_residual + hidden_states

        # --- FINAL SCALING ---
        hidden_states = hidden_states * self.layer_scalar

        # CRITICAL: return bare tensor, matching Gemma4TextDecoderLayer contract.
        return hidden_states
