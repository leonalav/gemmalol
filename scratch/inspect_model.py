from unsloth import FastLanguageModel
import torch
import sys
import os

# Mocking the model loading to see structure
model_id = "google/gemma-4-E2B-it"
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = 8192,
        load_in_4bit = False,
    )
    
    # Target text model
    base_model = getattr(model, 'base_model', model)
    if hasattr(base_model, 'model'):
        gemma4_model = base_model.model
    else:
        gemma4_model = base_model
    if hasattr(gemma4_model, 'language_model'):
        text_model = gemma4_model.language_model
    else:
        text_model = gemma4_model

    layer0 = text_model.layers[0]
    attn = layer0.self_attn
    print(f"Layer 0 Attention Type: {type(attn)}")
    print("Attributes of Layer 0 Attention:")
    for attr in dir(attn):
        if not attr.startswith('__'):
            val = getattr(attn, attr)
            if isinstance(val, torch.nn.Linear):
                print(f"  {attr}: Linear({val.in_features}, {val.out_features})")
                
except Exception as e:
    print(f"Error: {e}")
