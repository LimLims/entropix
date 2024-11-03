# File: entropix/download_weights.py

import os
import tyro
import torch
import ml_dtypes
import jax.numpy as jnp
from pathlib import Path

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def translate_key(in_key: str):
    """Translates HuggingFace model keys to SmolLM format."""
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
    elif out_key == 'lm_head':
        out_key = 'output'
    return f'{out_key}.weight'

def reverse_permute(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse permute operation for SmolLM attention weights."""
    # For SmolLM 1.7B, both query and key projections use same dimensions
    n_heads = 32  # SmolLM 1.7B uses 32 heads
    dim = 2048    # SmolLM 1.7B hidden dimension
    return tensor.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def main(model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", out_dir: Path = None):
    """Download and convert SmolLM weights."""
    if out_dir is None:
        out_dir = Path('weights/1.7B-Instruct')
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            offload_folder="/tmp/offload",
            device_map='cpu'
        )

        with torch.no_grad():
            state_dict = hf_model.state_dict()
            for hf_name, param in state_dict.items():
                print(f' {hf_name}: {param.shape=}')
                name = translate_key(hf_name)
                param = param.cpu()

                # Apply permutation for attention weights
                if name.endswith('wq.weight') or name.endswith('wk.weight'):
                    param = reverse_permute(param)

                # Convert to bfloat16 and save
                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                jnp.save(f'{out_dir}/{name}.npy', bf16_out)

if __name__ == "__main__":
    tyro.cli(main)