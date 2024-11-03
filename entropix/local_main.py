# File: entropix/local_main.py

import os
import time
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=true "
)

# Enable CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def get_device():
    """Get the first available GPU device or CPU if no GPU is available."""
    devices = jax.devices("gpu")
    if not devices:
        print("WARNING: No GPU devices found, falling back to CPU")
        return jax.devices("cpu")[0]
    return devices[0]

def to_device(x):
    """Helper function to move arrays to device."""
    return jax.device_put(x, get_device())

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

def build_attn_mask(seqlen: int, start_pos: int, max_seq_len: int = 2048) -> jax.Array:
    """Build an attention mask that allows attending to cached tokens and prevents attending to future tokens.
    
    Args:
        seqlen: Length of the current sequence
        start_pos: Number of cached tokens that can be attended to
        max_seq_len: Maximum sequence length (context window size)
        
    Returns:
        A mask of shape (1, 1, seqlen, max_seq_len) for broadcasting with attention scores
    """
    # Create mask for cached tokens (zeros)
    cache_mask = jnp.zeros((seqlen, start_pos), dtype=jnp.float32)
    
    # Create causal mask for current sequence
    curr_seq_mask = jnp.full((seqlen, seqlen), float("-inf"), dtype=jnp.float32)
    if seqlen > 1:
        curr_seq_mask = jnp.triu(curr_seq_mask, k=1)
    
    # Create mask for padding (all -inf)
    remaining_len = max_seq_len - (start_pos + seqlen)
    if remaining_len > 0:
        padding_mask = jnp.full((seqlen, remaining_len), float("-inf"), dtype=jnp.float32)
        
        # Combine all parts: [cache_tokens | current_sequence | padding]
        mask = jnp.concatenate([cache_mask, curr_seq_mask, padding_mask], axis=1)
    else:
        # Just combine cache and current sequence if no padding needed
        mask = jnp.concatenate([cache_mask, curr_seq_mask], axis=1)
    
    # Add broadcast dimensions to match scores shape (batch, heads, seq, seq)
    mask = mask[None, None, :, :]
    
    return mask

def __build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask



def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1.7B-Instruct')):
    # Print JAX configuration
    print("\nJAX Configuration:")
    print("==================")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Process index: {jax.process_index()}")
    print(f"Local device count: {jax.local_device_count()}")
    
    
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
    #xfmr_fn = jax.jit(xfmr, static_argnames=("model_params", "cur_pos"))

    #sample_fn = jax.jit(sample)



    print("\nCompiling functions...")
    start_time = time.time()

    # JIT compile the transformer and sampling functions
    xfmr_fn = jax.jit(
        xfmr, 
        static_argnames=("model_params", "cur_pos"),
        backend="gpu"
    )
    
    sample_fn = jax.jit(
        sample,
        backend="gpu"
    )
    
    print(f"Function compilation took: {time.time() - start_time:.2f} seconds")


    def generate(xfmr_weights, model_params, tokens):
        """Generate text from input tokens."""
        print("\nStarting generate function...")
        print("Moving input tensors to GPU...")
        
        gen_tokens = None
        cur_pos = 0
        
        # Explicitly move tensors to GPU and time the operations
        start_time = time.time()
        device = jax.devices("gpu")[0]
        
        # Move weights to GPU if needed
        print("Moving weights to GPU...")
        xfmr_weights = jax.tree_map(lambda x: jax.device_put(x, device), xfmr_weights)
        
        tokens = jax.device_put(jnp.array([tokens], jnp.int32), device)
        bsz, seqlen = tokens.shape
        
        attn_mask = jax.device_put(build_attn_mask(seqlen, cur_pos), device)
        freqs_cis = jax.device_put(
            precompute_freqs_cis(
                model_params.head_dim,
                model_params.max_seq_len,
                model_params.rope_theta,
                model_params.use_scaled_rope
            ),
            device
        )
        
        kvcache = KVCache.new(
            model_params.n_layers,
            bsz,
            model_params.max_seq_len,
            model_params.n_local_kv_heads,
            model_params.head_dim
        )
        kvcache = jax.tree_map(lambda x: jax.device_put(x, device), kvcache)
        
        print(f"Initial setup took: {time.time() - start_time:.2f} seconds")

        # Initial forward pass
        print("\nStarting first forward pass...")
        start_time = time.time()
        
        logits, kvcache, scores = xfmr_fn(
            xfmr_weights,
            model_params,
            tokens,
            cur_pos,
            freqs_cis[:seqlen],
            kvcache,
            attn_mask=attn_mask
        )
        jax.block_until_ready(logits)  # Make sure computation is complete
        print(f"First forward pass took: {time.time() - start_time:.2f} seconds")

        # Get first token
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        print(tokenizer.decode([next_token.item()]), end='', flush=True)

        cur_pos = seqlen
        stop = jnp.array([tokenizer.eos_id, tokenizer.eot_id, tokenizer.eom_id])
        sampler_cfg = SamplerConfig()
        gen_tokens = [next_token]

        # Generation loop
        print("\nStarting generation loop...")
        token_times = []
        
        while cur_pos < model_params.max_seq_len:
            token_start = time.time()
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
            jax.block_until_ready(next_token)  # Make sure computation is complete
            
            gen_tokens.append(next_token)
            token_time = time.time() - token_start
            token_times.append(token_time)
            
            # Decode and print token
            out_token = tokenizer.decode(next_token.tolist()[0])
            print(f"{out_token}", end='', flush=True)

            # Check if we hit a stop token
            if jnp.isin(next_token, stop).any():
                break
        
        avg_token_time = sum(token_times) / len(token_times)
        print(f"\n\nGeneration stats:")
        print(f"Average time per token: {avg_token_time:.3f} seconds")
        print(f"Tokens per second: {1/avg_token_time:.2f}")
    
        
    def __generate(xfmr_weights, model_params, tokens):
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



def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1.7B-Instruct')):
    """Main function for local inference."""
    # Detailed JAX configuration check
    print("\nJAX Configuration:")
    print("==================")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    print(f"Available GPU devices: {jax.devices('gpu')}")
    print(f"Available CPU devices: {jax.devices('cpu')}")
    
    # Test basic GPU operation
    print("\nTesting basic GPU operation:")
    try:
        x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
        device = jax.devices('gpu')[0]
        x = jax.device_put(x, device)
        start = time.time()
        y = jnp.dot(x, x.T)
        y.block_until_ready()
        end = time.time()
        print(f"Basic GPU matrix multiplication test took: {(end-start)*1000:.2f}ms")
        print(f"Test ran on device: {y.device()}")
    except Exception as e:
        print(f"GPU test failed with error: {e}")
        
    # Initialize model parameters
    model_params = SMOLLM_PARAMS

    # Load weights and initialize components
    print("\nLoading model...")
    xfmr_weights = load_weights(weights_path.absolute(), n_layers=model_params.n_layers)
    tokenizer = Tokenizer('tokenizer.json')

    # JIT compile the transformer function with explicit platforms
    print("\nCompiling functions...")
    print("Setting XLA flags for GPU compilation...")
    
    # Try to force GPU compilation
    platforms = jax.lib.xla_bridge.get_backend().platform
    print(f"XLA backend platform: {platforms}")
    
    xfmr_fn = jax.jit(
        xfmr, 
        static_argnames=("model_params", "cur_pos"),
        backend="gpu"
    )
    sample_fn = jax.jit(sample, backend="gpu")

    def generate(xfmr_weights, model_params, tokens):
        """Generate text from input tokens."""
        print("\nStarting generate function...")
        print("Moving input tensors to GPU...")
        
        gen_tokens = None
        cur_pos = 0
        
        # Explicitly move tensors to GPU and time the operations
        start_time = time.time()
        device = jax.devices("gpu")[0]
        
        # Move weights to GPU if needed
        print("Moving weights to GPU...")
        xfmr_weights = jax.tree_map(lambda x: jax.device_put(x, device), xfmr_weights)
        
        tokens = jax.device_put(jnp.array([tokens], jnp.int32), device)
        bsz, seqlen = tokens.shape
        
        attn_mask = jax.device_put(build_attn_mask(seqlen, cur_pos), device)
        freqs_cis = jax.device_put(
            precompute_freqs_cis(
                model_params.head_dim,
                model_params.max_seq_len,
                model_params.rope_theta,
                model_params.use_scaled_rope
            ),
            device
        )
        
        kvcache = KVCache.new(
            model_params.n_layers,
            bsz,
            model_params.max_seq_len,
            model_params.n_local_kv_heads,
            model_params.head_dim
        )
        kvcache = jax.tree_map(lambda x: jax.device_put(x, device), kvcache)
        
        print(f"Initial setup took: {time.time() - start_time:.2f} seconds")

        # Initial forward pass
        print("\nStarting first forward pass...")
        start_time = time.time()
        
        logits, kvcache, scores = xfmr_fn(
            xfmr_weights,
            model_params,
            tokens,
            cur_pos,
            freqs_cis[:seqlen],
            kvcache,
            attn_mask=attn_mask
        )
        jax.block_until_ready(logits)  # Make sure computation is complete
        print(f"First forward pass took: {time.time() - start_time:.2f} seconds")

        # Get first token
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        print(tokenizer.decode([next_token.item()]), end='', flush=True)

        cur_pos = seqlen
        stop = jnp.array([tokenizer.eos_id, tokenizer.eot_id, tokenizer.eom_id])
        sampler_cfg = SamplerConfig()
        gen_tokens = [next_token]

        # Generation loop
        print("\nStarting generation loop...")
        token_times = []
        
        while cur_pos < model_params.max_seq_len:
            token_start = time.time()
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
            jax.block_until_ready(next_token)  # Make sure computation is complete
            
            gen_tokens.append(next_token)
            token_time = time.time() - token_start
            token_times.append(token_time)
            
            # Decode and print token
            out_token = tokenizer.decode(next_token.tolist()[0])
            print(f"{out_token}", end='', flush=True)

            # Check if we hit a stop token
            if jnp.isin(next_token, stop).any():
                break
        
        avg_token_time = sum(token_times) / len(token_times)
        print(f"\n\nGeneration stats:")
        print(f"Average time per token: {avg_token_time:.3f} seconds")
        print(f"Tokens per second: {1/avg_token_time:.2f}")
    
    # Test tokenization
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>test<|eot_id|>"""
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    
    # Do a warmup pass to trigger compilation
    print("\nDoing warmup pass...")
    try:
        warmup_tokens = jax.device_put(jnp.array([tokens[:10]], jnp.int32), device)
        warmup_freqs = jax.device_put(
            precompute_freqs_cis(
                model_params.head_dim,
                10,
                model_params.rope_theta,
                model_params.use_scaled_rope
            ),
            device
        )
        warmup_kvcache = KVCache.new(
            model_params.n_layers,
            1,
            10,
            model_params.n_local_kv_heads,
            model_params.head_dim
        )
        warmup_mask = build_attn_mask(10, 0)
        
        start = time.time()
        logits, _, _ = xfmr_fn(
            xfmr_weights,
            model_params,
            warmup_tokens,
            0,
            warmup_freqs,
            warmup_kvcache,
            attn_mask=warmup_mask
        )
        logits.block_until_ready()
        print(f"Warmup pass took: {time.time() - start:.2f} seconds")
        print(f"Warmup ran on device: {logits.device()}")
    except Exception as e:
        print(f"Warmup failed with error: {e}")

    # Real generation
    print("\nStarting real generation...")
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokens)


if __name__ == '__main__':
    # Configure XLA flags
    
    
    tyro.cli(main)