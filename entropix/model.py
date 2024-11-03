# File: entropix/model.py

from typing import Optional, Tuple
import math
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add root to sys.path
from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.weights import XfmrWeights, LayerWeights

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-5) -> jax.Array:
    """RMS normalization with epsilon to prevent division by zero."""
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to query and key tensors."""
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis[None, :, None, :]
    xk_out = xk_ * freqs_cis[None, :, None, :]
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(dtype), xk_out.astype(dtype)

def attention(x: jax.Array, 
             layer_weights: LayerWeights, 
             model_params: ModelParams,
             cur_pos: int,
             layer_idx: int,
             freqs_cis: jax.Array,
             kvcache: KVCache,
             attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache, jax.Array]:
    """Multi-head attention mechanism."""
    # Check if x is 2D or 3D and adjust accordingly
    if x.ndim == 2:
        bsz = 1
        seq_len, dim = x.shape
        x = x[None, ...]  # Add batch dimension
    else:
        bsz, seq_len, dim = x.shape

    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

    # Linear projections
    xq = jnp.dot(x, layer_weights.wq.T).reshape(bsz, seq_len, model_params.n_local_heads, model_params.head_dim)
    xk = jnp.dot(x, layer_weights.wk.T).reshape(bsz, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xv = jnp.dot(x, layer_weights.wv.T).reshape(bsz, seq_len, model_params.n_local_kv_heads, model_params.head_dim)

    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)

    # Update KV cache
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

    # Reshape for attention computation
    xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = jnp.transpose(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)

    # Compute attention scores
    scores = jnp.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32

    # Debugging shapes
    #print(f"scores shape: {scores.shape}")        # Expected: (1,32,37,2048)
    
    #print(f"attn_mask shape: {attn_mask.shape}")  # Expected: (37,2048)

    def add_mask(_):
        return scores + attn_mask  # (1,32,37,2048) + (37,2048) -> Broadcastable

    def do_not_add_mask(_):
        return scores

    def conditional_add_mask(scores, attn_mask, cur_pos):
        # Define functions based on whether attn_mask is None
        def handle_none_mask(_):
            return lax.cond(cur_pos == 0, do_not_add_mask, lambda _: scores, operand=None)

        def handle_non_none_mask(_):
            return lax.cond(cur_pos == 0, add_mask, do_not_add_mask, operand=None)

        # Conditionally apply based on the presence of attn_mask
        return lax.cond(attn_mask is None, handle_none_mask, handle_non_none_mask, operand=None)

    # Example usage
    scores = conditional_add_mask(scores, attn_mask, cur_pos)


    # Apply masking
    mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)

    # Compute output
    output = jnp.matmul(scores.astype(values.dtype), values)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(bsz, seq_len, -1)
    out = jnp.dot(output, layer_weights.wo.T)

    # If input was 2D, remove the batch dimension from the output
    if x.ndim == 2:
        out = jnp.squeeze(out, axis=0)

    return out, kvcache, pre_scores

def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
    """Feed forward network with SwiGLU activation."""
    return jnp.dot(jax.nn.silu(jnp.dot(x, layer_weights.w1.T)) * 
                   jnp.dot(x, layer_weights.w3.T), layer_weights.w2.T)

def xfmr(xfmr_weights: XfmrWeights,
         model_params: ModelParams,
         tokens: jax.Array,
         cur_pos: int,
         freqs_cis: jax.Array,
         kvcache: KVCache,
         attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache, jax.Array]:
    """Main transformer model function."""
    # Token embeddings
    h = xfmr_weights.tok_embeddings[tokens]

    # Track attention scores for the last generated token
    latest_scores = None

    # Process layers
    for i in range(model_params.n_layers):
        # Attention
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(
            norm_x,
            xfmr_weights.layer_weights[i],
            model_params,
            cur_pos,
            i,
            freqs_cis,
            kvcache,
            attn_mask=attn_mask
        )
        if i == model_params.n_layers - 1:  # Save scores from last layer
            latest_scores = scores
        h = h + h_attn

        # Feed forward
        h = h + feed_forward(
            rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm),
            xfmr_weights.layer_weights[i]
        )

    # Final normalization and output projection
    logits = jnp.dot(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)

    return logits, kvcache, latest_scores