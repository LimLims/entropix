# File: entropix/weights.py
import torch
from typing import List, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils

from pathlib import Path
from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

class LayerWeights(NamedTuple):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    ffn_norm: jax.Array
    attention_norm: jax.Array

class XfmrWeights(NamedTuple):
    tok_embeddings: jax.Array
    norm: jax.Array
    output: jax.Array
    layer_weights: List[LayerWeights]

def create_partition_spec(key):
    dp = 'dp'
    mp = 'mp'
    fsdp = 'fsdp'
    if 'norm' in key:
        return PS()
    if 'rope.freqs' in key:
        return PS()
    elif 'tok_embeddings' in key or 'output' in key or 'w2' in key:
        return PS(fsdp, mp)
    else:
        return PS(mp, fsdp)

def translate_key(in_key: str):
    """Translates HuggingFace SmolLM keys to our format."""
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

def reverse_permute(tensor: jax.Array, config, is_kv: bool = False) -> jax.Array:
    """Reverse permute operation with adaptive dimensions based on model size and weight type."""
    # For SmolLM 1.7B, both query and key projections use the same dimensions
    n_heads = config["n_heads"]
    dim1 = config["dim"]
    
    return tensor.reshape(n_heads, 2, dim1 // n_heads // 2, config["dim"]).transpose(1, 2).reshape(dim1, config["dim"])

def fixed_get_imports(filename: str | Path) -> list[str]:
    """Workaround for HF modeling imports."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def download_weights(model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", out_dir: Path = None):
    """Download and process weights for SmolLM."""
    if out_dir is None:
        out_dir = Path('weights/1.7B-Instruct')
    out_dir.mkdir(parents=True, exist_ok=True)

    # SmolLM config 
    config = {
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 32,
        "vocab_size": 49152,
        "norm_eps": 1e-05,
        "rope_theta": 10000.0,
    }

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

                # Apply reverse permute for attention weights
                if name.endswith('wq.weight'):
                    param = reverse_permute(param, config, is_kv=False)
                elif name.endswith('wk.weight'):
                    param = reverse_permute(param, config, is_kv=True)

                # Convert to bfloat16 and save
                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                jnp.save(f'{out_dir}/{name}.npy', bf16_out)

        # Cleanup
        del hf_model
        del state_dict
        jax.clear_caches()


def load_weights(ckpt_dir: Path, n_layers: int = 24):  # Default to SmolLM layer count
    """Load SmolLM weights from checkpoint directory."""
    w = {}
    layer_weights = []

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("No GPU devices found")
    
    mesh = jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ('mp', 'fsdp'))
    
    for file in ckpt_dir.glob("*.npy"):
        name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
        weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
        partition_spec = create_partition_spec(name)
        sharding = NamedSharding(mesh, partition_spec)
        w[name] = jax.device_put(weight, sharding)
    
    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            wq=w[f'layers.{i}.attention.wq.weight'],
            wk=w[f'layers.{i}.attention.wk.weight'],
            wv=w[f'layers.{i}.attention.wv.weight'],
            wo=w[f'layers.{i}.attention.wo.weight'],
            w1=w[f'layers.{i}.feed_forward.w1.weight'],
            w2=w[f'layers.{i}.feed_forward.w2.weight'],
            w3=w[f'layers.{i}.feed_forward.w3.weight'],
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
            attention_norm=w[f'layers.{i}.attention_norm.weight'],
        ))
    
    xfmr_weights = XfmrWeights(
        tok_embeddings=w['tok_embeddings.weight'],
        norm=w['norm.weight'],
        output=w['output.weight'],
        layer_weights=layer_weights
    )

    return xfmr_weights