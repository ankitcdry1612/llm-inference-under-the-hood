"""
Milestone 3: The Manual Autoregressive Loop (No .generate() Magic)
==================================================================
Concept:
Modern developers think LLMs produce paragraphs of text in a single step.
Under the hood, an LLM is strictly AUTOREGRESSIVE:
  Input:  [Token_1, Token_2]                  --> Generates [Token_3]
  Input:  [Token_1, Token_2, Token_3]          --> Generates [Token_4]
  Input:  [Token_1, Token_2, Token_3, Token_4] --> Generates [Token_5]

In this script, you will learn:
1. How to implement the generation `while` loop completely from scratch.
2. How Greedy Decoding (temperature=0) vs Temperature Sampling works.
3. How to detect the End-of-Sequence (EOS) token to stop generation.
4. Real-time streaming of tokens to stdout as they are generated.
"""

import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

def manual_generate(
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    top_k: int = 50,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    print("=" * 70)
    print(" 🚀 MILESTONE 3: MANUAL AUTOREGRESSIVE GENERATION LOOP")
    print("=" * 70)
    print(f"\n[Config] Max Tokens: {max_new_tokens} | Temperature: {temperature} | Device: {device.upper()}")
    print(f'\n[Prompt]: "{prompt}"\n')
    print("[Streaming Generated Output]: ", end="", flush=True)

    # Encode initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_token_id = tokenizer.eos_token_id

    generated_tokens = []
    start_time = time.time()

    # THE CORE AUTOREGRESSIVE LOOP
    for step in range(max_new_tokens):
        with torch.no_grad():
            # 1. Forward pass with the current full context
            outputs = model(input_ids)
            
            # 2. Extract logits for the newest/last token position
            next_token_logits = outputs.logits[0, -1, :]

            # 3. Apply Temperature & Sampling Strategy
            if temperature == 0.0:
                # Greedy: Pick highest score directly (Argmax)
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
            else:
                # Temperature Scaling: Flatten or sharpen probability distribution
                scaled_logits = next_token_logits / temperature
                
                # Optional Top-K filtering: Keep only top K highest scores
                if top_k > 0:
                    top_values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    min_value = top_values[-1]
                    scaled_logits = torch.where(
                        scaled_logits < min_value,
                        torch.tensor(float("-inf")).to(device),
                        scaled_logits
                    )
                
                # Softmax -> Probabilities
                probs = torch.softmax(scaled_logits, dim=-1)
                # Sample from distribution
                next_token_id = torch.multinomial(probs, num_samples=1)

            token_val = next_token_id.item()
            generated_tokens.append(token_val)

            # 4. Check for End-of-Sequence (EOS) token
            if token_val == eos_token_id:
                print(" [EOS REACHED]")
                break

            # 5. Decode single token and stream to terminal immediately
            token_str = tokenizer.decode([token_val])
            sys.stdout.write(token_str)
            sys.stdout.flush()

            # 6. Append new token to input_ids for the NEXT iteration
            # Shape grows: [1, seq_len] -> [1, seq_len + 1]
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

    elapsed = time.time() - start_time
    total_generated = len(generated_tokens)
    tps = total_generated / elapsed if elapsed > 0 else 0

    print("\n\n" + "-" * 55)
    print(f"📊 Generation Stats:")
    print(f"   • Tokens Generated: {total_generated}")
    print(f"   • Time Taken:       {elapsed:.2f}s")
    print(f"   • Speed:            {tps:.2f} tokens/second")
    print("-" * 55)
    print("\n⚠️  NOTICE THE CRITICAL SYSTEMS FLAW IN THIS SCRIPT:")
    print("   At step N, we passed the ENTIRE sequence of (Prompt + N) tokens into the model.")
    print("   The model recomputed matrix math for ALL previous tokens on EVERY single step!")
    print("   Complexity: O(N^2) wasted compute. Milestone 4 fixes this via the KV-Cache.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_prompt = "The difference between Kafka and traditional message queues is that Kafka"
    manual_generate(test_prompt, max_new_tokens=40, temperature=0.7)
