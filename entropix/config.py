from typing import NamedTuple

class ModelConfig:
    """Configuration class for model parameters."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    max_seq_len: int

# Define SmolLM 1.7B configuration
SMOLLM_CONFIG = {
    "dim": 2048,
    "n_layers": 24,
    "n_heads": 32,
    "n_kv_heads": 32,
    "vocab_size": 49152,
    "norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,
    "max_seq_len": 2048,
}

class ModelParams(NamedTuple):
    n_layers: int 
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool

# Create SmolLM params instance
SMOLLM_PARAMS = ModelParams(
    n_layers=SMOLLM_CONFIG["n_layers"],
    n_local_heads=SMOLLM_CONFIG["n_heads"],
    n_local_kv_heads=SMOLLM_CONFIG["n_kv_heads"], 
    head_dim=SMOLLM_CONFIG["dim"] // SMOLLM_CONFIG["n_heads"],
    max_seq_len=SMOLLM_CONFIG["max_seq_len"],
    rope_theta=SMOLLM_CONFIG["rope_theta"],
    use_scaled_rope=SMOLLM_CONFIG["use_scaled_rope"]
)