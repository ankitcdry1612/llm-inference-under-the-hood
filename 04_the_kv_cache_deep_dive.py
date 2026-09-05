"""
Milestone 4: The KV-Cache Under The Hood
========================================
Concept:
In Milestone 3, we discovered the O(N^2) compute flaw: recalculating keys and values
for previous tokens on every step.

The KV-Cache solves this:
  - Cache the intermediate Key (K) and Value (V) tensors of past tokens in GPU/CPU RAM.
  - On Step N, pass ONLY the single newest token (Shape: [1, 1]) to the model!
  - The model computes K & V for the new token and appends it to `past_key_values`.

In this script, you will learn:
1. How to pass and maintain `past_key_values` (DynamicCache) in PyTorch.
2. The exact tensor dimensions of the KV-cache across all layers.
3. How KV-cache memory scales linearly with sequence length.
4. Benchmarking step-by-step latency with [1, 1] inputs.
"""

import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

def inspect_kv_cache_structure(past_key_values):
    """
    Inspects key and value cache tensors across Transformer layers.
    Handles DynamicCache (transformers >= 5.x / 4.40+) and legacy tuples.
    """
    if hasattr(past_key_values, "layers"):
        # Modern DynamicCache with DynamicLayer objects
        num_layers = len(past_key_values.layers)
        k_layer_0 = past_key_values.layers[0].keys
        v_layer_0 = past_key_values.layers[0].values
    elif hasattr(past_key_values, "key_cache"):
        # DynamicCache with key_cache list
        num_layers = len(past_key_values.key_cache)
        k_layer_0 = past_key_values.key_cache[0]
        v_layer_0 = past_key_values.value_cache[0]
    else:
        # Legacy tuple structure: ((k0, v0), (k1, v1), ...)
        num_layers = len(past_key_values)
        k_layer_0 = past_key_values[0][0]
        v_layer_0 = past_key_values[0][1]

    # Shape: [batch_size, num_heads, sequence_length, head_dim]
    batch_size, num_heads, seq_len, head_dim = k_layer_0.shape
    element_size_bytes = k_layer_0.element_size() # e.g. 2 bytes (fp16/bf16) or 4 bytes (fp32)
    
    # Total Elements = 2 (Key + Value) * num_layers * (batch * heads * seq_len * head_dim)
    total_elements = 2 * num_layers * (batch_size * num_heads * seq_len * head_dim)
    total_bytes = total_elements * element_size_bytes
    total_mb = total_bytes / (1024 * 1024)

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "total_mb": total_mb,
        "k_shape": list(k_layer_0.shape),
    }

def generate_with_kv_cache(prompt: str, max_new_tokens: int = 20):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    print("=" * 70)
    print(" 🚀 MILESTONE 4: THE KV-CACHE DEEP DIVE")
    print("=" * 70)
    print(f"\n[Prompt]: \"{prompt}\"")
    print(f"[Device]: {device.upper()}\n")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    
    # -------------------------------------------------------------
    # PHASE 1: PREFILL (Compute prompt & populate initial KV-cache)
    # -------------------------------------------------------------
    print(f"--- [PHASE 1: PREFILL] Processing Prompt ({prompt_len} tokens in parallel) ---")
    start_prefill = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    prefill_time = (time.time() - start_prefill) * 1000

    past_key_values = outputs.past_key_values
    stats = inspect_kv_cache_structure(past_key_values)
    print(f"  ✓ Prefill Time:               {prefill_time:.2f} ms")
    print(f"  ✓ Transformer Layers:         {stats['num_layers']}")
    print(f"  ✓ Attention Heads per layer:  {stats['num_heads']}")
    print(f"  ✓ Head Dimension:             {stats['head_dim']}")
    print(f"  ✓ Key Tensor Shape:           {stats['k_shape']} [batch, heads, seq_len, head_dim]")
    print(f"  ✓ KV-Cache Memory Footprint:  {stats['total_mb']:.4f} MB\n")

    # -------------------------------------------------------------
    # PHASE 2: DECODE LOOP (Generate 1 token at a time with KV-cache)
    # -------------------------------------------------------------
    print(f"--- [PHASE 2: DECODE] Generating Tokens using KV-Cache ---")
    print("Streaming: ", end="", flush=True)

    # For the first decode step, get the next token from prefill logits
    next_token_id = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
    sys.stdout.write(tokenizer.decode(next_token_id[0, 0]))
    sys.stdout.flush()

    eos_token_id = tokenizer.eos_token_id
    decode_times = []

    print("\n\nStep-by-Step Memory & Latency Tracking:")
    print("  " + "-" * 62)
    print(f"  | {'Step':<6} | {'Input Shape':<13} | {'KV Seq Len':<12} | {'KV Memory':<12} | {'Latency':<8} |")
    print("  " + "-" * 62)

    for step in range(1, max_new_tokens):
        t0 = time.time()
        
        with torch.no_grad():
            # NOTICE: We only pass `next_token_id` (Shape: [1, 1]) along with `past_key_values`!
            # We DO NOT pass the full sequence!
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        t1 = time.time()
        step_latency = (t1 - t0) * 1000
        decode_times.append(step_latency)

        # Update past_key_values
        past_key_values = outputs.past_key_values
        
        # Pick next token
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        token_val = next_token_id.item()
        
        stats = inspect_kv_cache_structure(past_key_values)
        print(f"  | #{step:<5} | {'[1, 1]':<13} | {stats['seq_len']:<12} | {stats['total_mb']:.4f} MB    | {step_latency:>6.2f}ms |")

        if token_val == eos_token_id:
            print("  [EOS Reached]")
            break

    print("  " + "-" * 62)
    avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
    print(f"\n📊 Summary Metrics:")
    print(f"   • Avg Decode Latency per Token: {avg_decode:.2f} ms")
    print(f"   • Input shape per step was always [1, 1] instead of growing!")
    print(f"   • KV-Cache avoided recomputing Attention for all past tokens.")
    print("\n💡 The Systems Bottleneck:")
    print("   Look at the 'KV Memory' column: it grows on EVERY single token.")
    print("   Because pre-2023 systems allocated contiguous memory for max_len (e.g. 2048),")
    print("   they wasted 70%+ of VRAM. Milestone 5 simulates PagedAttention to solve this!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    generate_with_kv_cache("A distributed key-value store like Redis manages memory by", max_new_tokens=15)
