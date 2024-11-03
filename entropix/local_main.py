# File: entropix/local_main.py

from typing import Tuple
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro
import torch  # Required for download_weights

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add root to sys.path
from entropix.config import SMOLLM_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights, download_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def apply_scaling(freqs: jax.Array):
    """Scale frequencies for rotary embeddings."""
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 2048  # SmolLM context length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    return jax.vmap(scale_freq)(freqs)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Precompute frequency cis for rotary embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def build_attn_mask(self, seqlen: int, cur_pos: int) -> jax.Array:
    key_len = self.params.max_seq_len  # 2048
    
    # Initialize a mask with -inf
    mask = jnp.full((seqlen, key_len), float("-inf"), dtype=jnp.float32)
    
    if seqlen > 1:
        # Create an upper triangular matrix for the current sequence
        seqlen_mask = jnp.triu(jnp.ones((seqlen, seqlen), dtype=jnp.float32) * float("-inf"), k=1)
        
        # Concatenate zeros for the cached tokens
        # Assuming cached tokens are in the first (key_len - seqlen) positions
        cache_mask = jnp.zeros((seqlen, key_len - seqlen), dtype=jnp.float32)
        
        # Combine the masks
        mask = jnp.concatenate([cache_mask, seqlen_mask], axis=1)
    
    return mask

def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1.7B-Instruct')):
    """Main function for local inference."""
    # Initialize model parameters
    model_params = SMOLLM_PARAMS

    # First download weights if they don't exist
    if not weights_path.exists():
        print(f"Downloading weights to {weights_path}...")
        download_weights(out_dir=weights_path)
    
    # Load weights and initialize components
    xfmr_weights = load_weights(weights_path.absolute(), n_layers=model_params.n_layers)
    tokenizer = Tokenizer('tokenizer.json')
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params", "cur_pos"))

    sample_fn = jax.jit(sample)

    def generate(xfmr_weights, model_params, tokens):
        """Generate text from input tokens."""
        gen_tokens = None
        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(
            model_params.head_dim,
            model_params.max_seq_len,
            model_params.rope_theta,
            model_params.use_scaled_rope
        )
        kvcache = KVCache.new(
            model_params.n_layers,
            bsz,
            model_params.max_seq_len,
            model_params.n_local_kv_heads,
            model_params.head_dim
        )

        # Initial forward pass
        logits, kvcache, scores, = xfmr_fn(
            xfmr_weights,
            model_params,
            tokens,
            cur_pos,
            freqs_cis[:seqlen],
            kvcache,
            attn_mask=attn_mask
        )

        # Get first token
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        print(tokenizer.decode([next_token.item()]), end='', flush=True)

        cur_pos = seqlen
        stop = jnp.array([tokenizer.eos_id, tokenizer.eot_id, tokenizer.eom_id])
        sampler_cfg = SamplerConfig()
        gen_tokens = [next_token]

        # Generation loop
        while cur_pos < model_params.max_seq_len:
            cur_pos += 1
            logits, kvcache, scores = xfmr_fn(
                xfmr_weights,
                model_params,
                next_token,
                cur_pos,
                freqs_cis[cur_pos:cur_pos+1],
                kvcache
            )
            next_token = sample_fn(logits, scores, gen_tokens)
            gen_tokens.append(next_token)
            
            # Decode and print token
            out_token = tokenizer.decode(next_token.tolist()[0])
            print(out_token, end='', flush=True)

            # Check if we hit a stop token
            if jnp.isin(next_token, stop).any():
                break

    # Example prompt
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a world-class AI system, capable of complex reasoning and reflection.<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you help me understand how neural networks learn?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print(prompt)
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokens)

if __name__ == '__main__':
    # Configure XLA flags
    import os
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
    )
    
    tyro.cli(main)