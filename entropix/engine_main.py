import asyncio
from pathlib import Path

import jax
import tyro

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add root to sys.path
from entropix.engine import SMOLLM_PARAMS, EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights, download_weights

class Metadata:
    def __init__(self):
        self.start_time = None

class Request:
    def __init__(
        self,
        tokens: jax.Array,
        max_tokens: int,
        metadata: Metadata,
        is_client_side_tokenization: bool = False,
    ):
        self.tokens: jax.Array = tokens
        self.max_tokens: int = max_tokens
        self.metadata: Metadata = metadata
        self.is_client_side_tokenization: bool = is_client_side_tokenization

async def run(
    ckpt_path: Path = Path("weights/1.7B-Instruct"),
    tokenizer_path: str = "tokenizer.json"
):
    """Run the SmolLM engine."""
    model_params = SMOLLM_PARAMS

    # Download weights if they don't exist
    if not ckpt_path.exists():
        print(f"Downloading weights to {ckpt_path}...")
        download_weights(out_dir=ckpt_path)

    xfmr_weights = load_weights(ckpt_path, n_layers=model_params.n_layers)
    tokenizer = Tokenizer(tokenizer_path)
    
    # JIT compile functions
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params", "cur_pos"))

    sample_fn = jax.jit(sample)

    num_engines = 1
    driver = Driver(
        prefill_engines=[
            EntropixEngine(
                model_params,
                xfmr_weights,
                tokenizer,
                xfmr_fn,
                sample_fn
            )
            for _ in range(num_engines)
        ],
        generate_engines=[
            EntropixEngine(
                model_params,
                xfmr_weights,
                tokenizer,
                xfmr_fn,
                sample_fn
            )
            for _ in range(num_engines)
        ],
        prefill_params=[model_params] * num_engines,
        generate_params=[model_params] * num_engines,
    )

    orchestrator = EntropixOrchestrator(driver)

    # Example prompt
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

Think carefully in a step-by-step manner. Can you explain how neural networks learn?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # Create multiple concurrent requests
    requests = [
        Request(
            tokens=prompt,
            max_tokens=2048,
            metadata=Metadata()
        )
        for _ in range(4)
    ]

    async def process_generator(gen, loop_num):
        async for decoded in gen:
            print(f"LOOP {loop_num}: {decoded}")

    # Process requests concurrently
    generators = [orchestrator.decode(request) for request in requests]
    await asyncio.gather(
        *[process_generator(gen, i + 1) for i, gen in enumerate(generators)]
    )

def main():
    """Entry point."""
    asyncio.run(run())

if __name__ == "__main__":
    # Configure XLA flags
    import os
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )

    tyro.cli(main)