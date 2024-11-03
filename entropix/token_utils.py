# File: entropix/token_utils.py

from bisect import bisect_left
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from entropix.tokenizer import Tokenizer

ResultTokens = Any

@dataclasses.dataclass
class ReturnSample:
    """Both the token ids and their string representation."""
    text: list[str]
    token_ids: list[int]

DEFAULT_PREFILL_BUCKETS = [
    16, 32, 64, 128, 256, 512, 1024, 2048
]

def take_nearest_length(lengths: list[int], length: int) -> int:
    """Gets the nearest length to the right in a set of lengths."""
    pos = bisect_left(lengths, length)
    if pos == len(lengths):
        return lengths[-1]
    return lengths[pos]

def tokenize_and_pad(
    s: str,
    vocab,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Tokenize and pad a string."""
    tokens = np.array(vocab.encode_tf(s))
    bos_id = vocab.bos_id
    pad_id = vocab.pad_id
    assert pad_id == 0, "Further logic required if pad_id not 0."

    return pad_tokens(
        tokens=tokens,
        bos_id=bos_id,
        pad_id=pad_id,
        is_bos=is_bos,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=jax_padding,
    )

def pad_tokens(
    tokens: np.ndarray,
    bos_id: int,
    pad_id: int,
    is_bos: bool = True,
    prefill_lengths: Optional[List[int]] = None,
    max_prefill_length: Optional[int] = None,
    jax_padding: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Pad tokens to nearest prefill length."""
    if prefill_lengths is None:
        prefill_lengths = DEFAULT_PREFILL_BUCKETS
    if max_prefill_length is not None:
        prefill_lengths = prefill_lengths[:prefill_lengths.index(max_prefill_length)] + [max_prefill_length]

    if is_bos:
        tokens = np.concatenate([np.array([bos_id]), tokens], axis=-1)
    
    true_length = tokens.shape[-1]
    padded_length = take_nearest_length(prefill_lengths, true_length)
    padding = padded_length - true_length

    if padding < 0:
        logging.warning("Provided sequence longer than available.")
        padded_tokens = tokens[-padded_length:]
    else:
        padded_tokens = np.pad(tokens, (0, padding), constant_values=(pad_id,))
        
    if jax_padding:
        padded_tokens = jnp.array(padded_tokens)
        
    return padded_tokens, true_length

def process_result_tokens(
    tokenizer: Tokenizer,
    slot: int,
    slot_max_length: int,
    result_tokens: ResultTokens,
    complete: np.ndarray,
    is_client_side_tokenization: bool = False,
    debug: bool = False,
) -> Tuple[List[ReturnSample], np.ndarray]:
    """Process result tokens into a list of strings."""
    slot_data = result_tokens.get_result_at_slot(slot)
    slot_tokens = slot_data.tokens
    slot_valid = slot_data.valid
    slot_lengths = slot_data.lengths
    samples, speculations = slot_tokens.shape

    complete = complete | (slot_lengths > slot_max_length)
    
    if debug:
        logging.info(
            "Complete %s, slot_tokens: %s, slot_lengths: %s",
            str(complete),
            str(slot_tokens),
            str(slot_lengths),
        )

    return_samples = []
    for idx in range(samples):
        text_so_far = []
        tok_id_so_far = []
        if not complete[idx].item():
            for spec_idx in range(speculations):
                tok_id = slot_tokens[idx, spec_idx].item()
                valid = slot_valid[idx, spec_idx].item()
                if debug:
                    logging.info(
                        "Sample idx: %d Speculation idx: %d Token: %d",
                        idx,
                        spec_idx,
                        tok_id,
                    )
                if tok_id in tokenizer.stop_tokens or not valid:
                    complete[idx] = True
                    tok_id_so_far.append(tok_id)
                    break
                else:
                    if not is_client_side_tokenization:
                        text_so_far.append(tokenizer.decode([tok_id]))
                    tok_id_so_far.append(tok_id)
                    
        return_samples.append(
            ReturnSample(text=text_so_far, token_ids=tok_id_so_far)
        )
        if debug:
            logging.info("Return samples %s", str(return_samples))
            
    return return_samples, complete

def is_byte_token(s: str) -> bool:
    """Returns True if s is a byte string like "<0xAB>"."""
    if len(s) != 6 or s[0:3] != "<0x" or s[-1] != ">":
        return False
    return True

def text_tokens_to_str(text_tokens: Iterable[str]) -> str:
    """Converts token text to a single string, collapsing bytes."""
    bytes_so_far = []
    for text_token in text_tokens:
        if is_byte_token(text_token):
            bytes_so_far.append(bytes([int(text_token[1:-1], 16)]))
        else:
            bytes_so_far.append(bytes(text_token, "utf-8"))
    return b"".join(bytes_so_far).decode("utf-8", "replace")