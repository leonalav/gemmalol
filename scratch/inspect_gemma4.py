import torch
import sys
import os

# Path fixing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'src' else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, Gemma4TextConfig
    config = Gemma4TextConfig()
    attn = Gemma4TextAttention(config, layer_idx=0)
    print("Attributes of Gemma4TextAttention:")
    for attr in dir(attn):
        if not attr.startswith('__'):
            val = getattr(attn, attr)
            if isinstance(val, torch.nn.Linear):
                print(f"  {attr}: Linear({val.in_features}, {val.out_features})")
except Exception as e:
    print(f"Error: {e}")
