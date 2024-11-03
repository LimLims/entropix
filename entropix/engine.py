# File: entropix/engine.py

import functools
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as PS

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add root to sys.path
from entropix.kvcache import KVCache
from entropix.tokenizer import Tokenizer
from entropix.config import SMOLLM_CONFIG, ModelConfig

"""Defines the Entropix Engine API for SmolLM inference."""

# Type definitions
Params = Any
Prefix = Any
DeviceTokens = Any
CpuDevices = Any

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

class DecodeState(NamedTuple):
    """The inputs into a generation step."""
    prefill_cache: jax.Array
    generate_cache: jax.Array
    generate_cache_index: int
    generate_lengths: jax.Array
    generated_token: jax.Array

class SlotData(NamedTuple):
    """Class to store slot data."""
    tokens: Union[jax.Array, np.ndarray]
    valid: Union[jax.Array, np.ndarray]
    lengths: Union[jax.Array, np.ndarray]

class ResultTokens(NamedTuple):
    """Class to store returned tokens."""
    data: Union[jax.Array, np.ndarray]
    tokens_idx: Tuple[int, int]
    valid_idx: Tuple[int, int]
    length_idx: Tuple[int, int]
    samples_per_slot: int

    def copy_to_host_async(self: "ResultTokens") -> None:
        if isinstance(self.data, np.ndarray):
            return
        self.data.copy_to_host_async()

    def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
        return ResultTokens(
            np.array(self.data),
            self.tokens_idx,
            self.valid_idx,
            self.length_idx,
            self.samples_per_slot,
        )

    def get_result_at_slot(self, slot: int) -> SlotData:
        start_idx = slot * self.samples_per_slot
        end_idx = (slot + 1) * self.samples_per_slot
        return SlotData(
            tokens=self.data[start_idx:end_idx, self.tokens_idx[0]:self.tokens_idx[1]],
            valid=self.data[start_idx:end_idx, self.valid_idx[0]:self.valid_idx[1]],
            lengths=self.data[start_idx:end_idx, self.length_idx[0]:self.length_idx[1]][:, 0],
        )

class EntropixEngine:
    def __init__(
        self,
        params: Params,
        xfmr_weights: XfmrWeights,
        tokenizer: Tokenizer,
        xfmr_fn: Callable,
        sample_fn: Callable,
    ):
        self.params = params
        self.xfmr_weights = xfmr_weights
        self.tokenizer = tokenizer
        self.xfmr_fn = xfmr_fn
        self.sample_fn = sample_fn
        self.freqs_cis = self.precompute_freqs_cis(
            params.head_dim,
            params.max_seq_len,
            params.rope_theta,
            params.use_scaled_rope
        )

    def get_prefix_destination_sharding(self) -> Any:
        return None

    def init_decode_state(self, *args, **kwargs) -> DecodeState:
        return DecodeState(
            prefill_cache=None,
            generate_cache=None,
            generate_cache_index=0,
            generate_lengths=jnp.zeros((1,), dtype=jnp.int32),
            generated_token=jnp.zeros((1, 1), dtype=jnp.int32)
        )

    def get_tokenizer(self) -> Dict[str, Any]:
        return {}

    def build_tokenizer(self, metadata: Dict[str, Any]) -> Tokenizer:
        return self.tokenizer

    @property
    def max_concurrent_decodes(self) -> int:
        return 1

    @property
    def samples_per_slot(self) -> int:
        return 1

    def free_resource(self, slot: int) -> Any:
        return None

    @property
    def max_prefill_length(self) -> int:
        return self.params.max_seq_len

    def apply_scaling(self, freqs: jax.Array):
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
                lambda _: jax.lax.cond(
                    wavelen > low_freq_wavelen,
                    lambda _: freq / SCALE_FACTOR,
                    scale_mid,
                    None
                ),
                None
            )

        return jax.vmap(scale_freq)(freqs)

    def precompute_freqs_cis(
        self,
        dim: int,
        end: int,
        theta: float = 10000.0,
        use_scaled: bool = False,
        dtype: jnp.dtype = jnp.float32
    ) -> jax.Array:
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
        if use_scaled:
            freqs = self.apply_scaling(freqs)
        t = jnp.arange(end, dtype=dtype)
        freqs = jnp.outer(t, freqs)
        return jnp.exp(1j * freqs)

    def build_attn_mask(self, seqlen: int, start_pos: int) -> jax.Array:
        mask = None
        if seqlen > 1:
            mask = jnp.full((seqlen, seqlen), float("-inf"))
            mask = jnp.triu(mask, k=1)
            mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask])
        return mask.astype(jnp.float32) if mask is not None else jnp.zeros((seqlen, seqlen), dtype=jnp.float32)

    @functools.partial(jax.jit, static_argnames=("self", "params"))
    def prefill(
        self,
        *,
        params: Params,
        existing_prefix: Optional[jax.Array] = None,
        padded_tokens: jax.Array, 
        true_length: int,
        sampler: Optional[Callable[[Any], Any]] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[Prefix, ResultTokens]:
        """Compute KV cache for input tokens."""
        cur_pos = 0
        bsz, seqlen = padded_tokens.shape
        attn_mask = self.build_attn_mask(seqlen, cur_pos)
        
        kvcache = KVCache.new(
            params.n_layers,
            bsz,
            params.max_seq_len,
            params.n_local_kv_heads,
            params.head_dim
        )

        # Initial forward pass
        logits, kvcache, scores = self.xfmr_fn(
            self.xfmr_weights,
            params,
            padded_tokens,
            cur_pos,
            self.freqs_cis[:seqlen],
            kvcache,
            attn_mask=attn_mask
        )

        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

        # Package results
        tokens = next_token
        validity = jnp.ones_like(next_token, dtype=jnp.bool_)
        lengths = jnp.array([[true_length + 1]], dtype=jnp.int32)
        
        data = jnp.concatenate([tokens, validity, lengths], axis=1)
        
        result = ResultTokens(
            data=data,
            tokens_idx=(0, 1),
            valid_idx=(1, 2),
            length_idx=(2, 3),
            samples_per_slot=1,
        )

        prefill_result = {
            "logits": logits,
            "cache": kvcache,
            "next_pos": seqlen,
            "generated_tokens": jnp.zeros((bsz, 1), dtype=jnp.int32),
            "tokens": next_token,
        }

        return prefill_result, result
    

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def generate(
        self,
        params: Params,
        decode_state: DecodeState,
        sampler: Optional[Callable[[Any], Any]] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> Tuple[DecodeState, ResultTokens]:
        """Generate next token given current state."""
        cur_pos = decode_state["next_pos"]
        freqs_cis_slice = jax.lax.dynamic_slice(
            self.freqs_cis,
            (cur_pos, 0),
            (1, self.freqs_cis.shape[1])
        )

        # Forward pass
        logits, kvcache, scores = self.xfmr_fn(
            self.xfmr_weights,
            params,
            decode_state["tokens"],
            cur_pos,
            freqs_cis_slice,
            decode_state["cache"],
        )

        # Sample next token
        new_token = self.sample_fn(logits, scores)

        # Package results
        result = ResultTokens(
            data=jnp.concatenate(
                (
                    new_token,
                    jnp.ones_like(new_token, dtype=jnp.bool_),
                    decode_state["generated_tokens"],
                ),
                axis=1,
            ),
            tokens_idx=(0, 1),
            valid_idx=(1, 2),
            length_idx=(2, 3),
            samples_per_slot=1,
        )

        new_state = {
            "logits": logits,
            "cache": kvcache,
            "next_pos": decode_state["next_pos"] + 1,
            "generated_tokens": decode_state["generated_tokens"] + 1,
            "tokens": new_token,
        }

        return new_state, result

    @functools.partial(
        jax.jit,
        static_argnums=(0,),
        donate_argnums=(1, 2,),
    )
    def insert(
        self,
        prefix: Prefix,
        decode_state: DecodeState,
        slot: int,
    ) -> DecodeState:
        """Insert new request into existing decode state at specified slot."""
        return {
            "logits": prefix["logits"],
            "cache": prefix["cache"],
            "next_pos": prefix["next_pos"],
            "generated_tokens": prefix["generated_tokens"],
            "tokens": prefix["tokens"],
        }

    @property
    def mesh(self) -> jax.sharding.Mesh:
        """Return the mesh for model parallelism."""
        return None

    @property
    def colocated_cpus(self) -> Union[list[CpuDevices], None]:
        """Return colocated CPU devices."""
        return None