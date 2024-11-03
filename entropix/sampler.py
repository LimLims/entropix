from dataclasses import dataclass
from typing import Dict, Tuple
import jax
import jax.numpy as jnp

MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


@dataclass
class SamplerConfig:
    # Base sampling parameters  
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_probability: float = 0.03

    # Entropy thresholds
    low_logits_entropy_threshold: float = 0.5
    medium_logits_entropy_threshold: float = 1.484 
    high_logits_entropy_threshold: float = 2.07

    # Varentropy thresholds
    low_logits_varentropy_threshold: float = 3.18
    medium_logits_varentropy_threshold: float = 3.75
    high_logits_varentropy_threshold: float = 6.08

    # Attention thresholds
    low_attention_entropy_threshold: float = 7.34
    medium_attention_entropy_threshold: float = 7.78
    high_attention_entropy_threshold: float = 8.05
    
    low_attention_varentropy_threshold: float = 5.112
    medium_attention_varentropy_threshold: float = 5.8125
    high_attention_varentropy_threshold: float = 6.82

    # New parameters from notebook
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2
    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5

    # Adaptive sampling parameters
    number_of_adaptive_samples: int = 5
    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2
    adaptive_top_p_coefficient: float = 0.1
    adaptive_min_p_coefficient: float = 0.5
    
    # Scoring coefficients
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4


def sample(
    logits: jax.Array,
    attention_scores: jax.Array,
    gen_tokens: jax.Array,
    clarifying_question_token: int = 2564,
    key=jax.random.PRNGKey(1337),
) -> Tuple[jax.Array, str]:
    cfg = SamplerConfig()
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]

    def _and(*args):
        res = True
        for a in args:
            res = jax.lax.bitwise_and(res, a)
        return res

    # State detection conditions
    FLOWING = _and(
        ent < cfg.low_logits_entropy_threshold,
        vent < cfg.low_logits_varentropy_threshold,
        attn_ent < cfg.low_attention_entropy_threshold,
        attn_vent < cfg.low_attention_varentropy_threshold,
    ).astype(float)

    TREADING = _and(
        ent > cfg.high_logits_entropy_threshold,
        vent < cfg.low_logits_varentropy_threshold,
        attn_ent < cfg.low_attention_entropy_threshold,
        attn_vent < cfg.low_attention_varentropy_threshold,
    ).astype(float)

    EXPLORING = _and(
        ent < cfg.high_logits_entropy_threshold,
        vent > cfg.high_logits_varentropy_threshold,
        attn_ent < cfg.low_attention_entropy_threshold,
        attn_vent > cfg.high_attention_varentropy_threshold,
    ).astype(float)

    RESAMPLING = _and(
        ent > cfg.medium_logits_entropy_threshold,
        vent > cfg.high_logits_varentropy_threshold,
        attn_ent > cfg.high_attention_entropy_threshold,
        attn_vent > cfg.high_attention_varentropy_threshold,
    ).astype(float)

    case = jnp.argmax(jnp.hstack([FLOWING, TREADING, EXPLORING, RESAMPLING, jnp.array(1.0).reshape(1)]))

    def flowing():
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    def treading():
        if not jnp.isin(gen_tokens[:, -1], jnp.array([clarifying_question_token])).any():
            return jnp.array([[clarifying_question_token]])
        temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent
        return _sample(
            logits,
            temperature=jnp.minimum(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key=key
        )

    def exploring():
        temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent
        return _sample(
            logits,
            temperature=jnp.minimum(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key=key
        )

    def resampling():
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent
        top_p_adj = jnp.maximum(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attn_ent)
        return _sample(
            logits,
            temperature=jnp.minimum(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key=key
        )

    def adaptive_sampling():
        def score_sample(sample):
            sample_oh = jax.nn.one_hot(sample, logits.shape[-1])
            log_prob = jnp.sum(jax.nn.log_softmax(logits[:, -1]) * sample_oh, axis=-1)
            confidence_score = (
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient
            )
            return log_prob + confidence_score

        keys = jax.random.split(key, cfg.number_of_adaptive_samples)
        samples = []
        for sample_key in keys:
            sample = _sample(
                logits,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_probability,
                key=sample_key
            )
            samples.append(sample)

        sample_scores = jnp.array([score_sample(sample) for sample in samples])
        best_sample_idx = jnp.argmax(sample_scores)
        return samples[best_sample_idx]

    return jax.lax.switch(case, (flowing, treading, exploring, resampling, adaptive_sampling))


def calculate_varentropy_logsoftmax(
  logits: jnp.ndarray, axis: int = -1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
  log_probs = jax.nn.log_softmax(logits, axis=axis)
  probs = jnp.exp(log_probs)
  entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
  varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None]) ** 2, axis=axis)
  return entropy, varentropy


def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
  """Samples one token from a multinomial distribution with sorted probabilities."""
  q = jax.random.exponential(key=key, shape=probs_sort.shape)
  return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)


def nucleus_sample(
  logits: jax.Array,
  temperature=0.666,
  top_p=0.90,
  top_k=27,
  key=jax.random.PRNGKey(1337),
) -> jax.Array:
  bsz = logits.shape[0]
  logit = logits[:, -1]
  probs = jax.nn.softmax(logit / temperature, axis=-1)

  # Apply top-k sampling
  top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
  probs_sort_jax = jnp.flip(top_k_probs, axis=-1)
  probs_idx_jax = jnp.flip(top_k_indices, axis=-1)
  probs_sum_jax = jnp.cumsum(probs_sort_jax, axis=-1)

  # Apply top-p sampling
  mask_jax = jnp.where(
    probs_sum_jax - probs_sort_jax > top_p, True, False
  )  # Use jnp.where
  probs_sort_jax = probs_sort_jax * (
    1 - mask_jax
  )  # Set values to 0.0 using multiplication
  probs_sort_jax = probs_sort_jax / jnp.sum(probs_sort_jax, axis=-1, keepdims=True)

  next_token_jax = multinomial_sample_one(probs_sort_jax, key)
  next_token_g_jax = jnp.take_along_axis(
    probs_idx_jax, next_token_jax.reshape(bsz, 1), axis=-1
  )
  return next_token_g_jax.astype(jnp.int32)


def _sample(
  logits: jax.Array,
  *,
  temperature: float | jax.Array,
  top_p: float | jax.Array,
  top_k: int | jax.Array,
  min_p: float | jax.Array,
  key=jax.random.PRNGKey(1337),
) -> jax.Array:
  bsz = logits.shape[0]
  logit = logits[:, -1]
  probs = jax.nn.softmax(logit / temperature, axis=-1)

  # Maybe apply min_p sampling
  p_max = jnp.max(probs, axis=-1, keepdims=True)
  indices_to_remove = probs < (min_p * p_max)
  min_p_sampled_logit = jnp.where(
    indices_to_remove, jnp.full_like(logit, float("-inf")), logit
  )
  logit = jax.lax.select(min_p > 0.0, min_p_sampled_logit, logit)

  # Apply top-k sampling
  iota = jax.lax.iota(jnp.int32, probs.shape[1]).reshape(probs.shape)
  _top_k_probs, _top_k_indices = jax.lax.sort_key_val(probs, iota)
  probs_sort = jnp.where(
    jnp.flip(iota[:, :MAX_K], axis=-1) < top_k, _top_k_probs[:, -MAX_K:], 0.0
  )
  probs_idx = _top_k_indices[:, -MAX_K:]

  probs_sum = jnp.cumsum(probs_sort, axis=-1)
  # Apply top-p sampling
  mask = jnp.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
  probs_sort = probs_sort * (1 - mask)
  probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
  next_token = multinomial_sample_one(probs_sort, key)
  next_token_g = jnp.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
  return next_token_g.astype(jnp.int32)


def calculate_metrics(
  logits: jnp.ndarray, attention_scores: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
  entropy, varentropy = calculate_varentropy_logsoftmax(logits)
  attention_probs = jax.nn.softmax(attention_scores, axis=-1)
  attn_entropy = -jnp.sum(
    attention_probs * jnp.log2(jnp.clip(attention_probs, 1e-10, 1.0)), axis=-1
  )
  attn_varentropy = jnp.var(attn_entropy, axis=1)

  return {
    "logits_entropy": jnp.mean(entropy),
    "logits_varentropy": jnp.mean(varentropy),
    "attn_entropy": jnp.mean(attn_entropy),
    "attn_varentropy": jnp.mean(attn_varentropy),
  }