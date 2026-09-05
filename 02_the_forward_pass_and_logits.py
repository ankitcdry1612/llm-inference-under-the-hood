"""
Milestone 2: The Raw Forward Pass & Logits
==========================================
Concept:
When token IDs enter an LLM, the model processes them through Transformer layers
(Self-Attention + Feed Forward Networks) and projects the result onto the final
Language Model Head (LM Head).

The output for the final token position is a vector of numbers called LOGITS:
  Logits = [Raw unnormalized scores for every single token in the vocabulary (~49k tokens)]

In this script, you will learn:
1. What tensor shapes enter and leave the model.
2. What "Logits" actually look like in memory.
3. How to use Softmax to turn raw logits into clean probabilities (summing to 100%).
4. How to inspect Top-K candidate predictions for the next token.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

def run_forward_pass_demo():
    print("=" * 70)
    print(" 🚀 MILESTONE 2: THE RAW FORWARD PASS & LOGITS")
    print("=" * 70)

    # Detect Apple Silicon (MPS) or CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[1] Loading Model & Tokenizer on device: [{device.upper()}]...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval() # Set to inference mode (disable dropout)

    # 1. Prepare input
    prompt = "The primary purpose of an Operating System is to"
    print(f'\n[2] Input Prompt: "{prompt}"')
    
    # input_ids shape: [Batch_Size=1, Sequence_Length]
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    batch_size, seq_len = input_ids.shape
    print(f"    Input Tensor Shape: [batch_size={batch_size}, seq_len={seq_len}]")

    # 2. Execute a SINGLE forward pass (No generation loop)
    print("\n[3] Executing single forward pass through Transformer layers...")
    with torch.no_grad():
        outputs = model(input_ids)
    
    # 3. Inspect Logits
    # Shape: [batch_size, seq_len, vocab_size]
    logits = outputs.logits
    print(f"    Raw Output Logits Shape: {list(logits.shape)}")
    print(f"    -> batch_size: {logits.shape[0]}")
    print(f"    -> sequence_len: {logits.shape[1]} (Logits computed for EVERY input position)")
    print(f"    -> vocab_size: {logits.shape[2]:,} (One score for each word in vocabulary)")

    # 4. Extract logits for the LAST token position (to predict the next token)
    next_token_logits = logits[0, -1, :]
    print(f"\n[4] Last Token Position Logits Vector:")
    print(f"    Shape: {list(next_token_logits.shape)}")
    print(f"    Sample raw scores (first 5 tokens): {next_token_logits[:5].tolist()}")
    print(f"    Min logit: {next_token_logits.min().item():.2f} | Max logit: {next_token_logits.max().item():.2f}")

    # 5. Softmax: Convert raw logits to Probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)
    print(f"\n[5] Converted to Probabilities via Softmax:")
    print(f"    Sum of all probabilities: {probabilities.sum().item():.4f} (Must equal 1.0)")

    # 6. Top-5 Predictions
    top_k = 5
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    
    print(f"\n[6] Top {top_k} Most Probable Next Tokens Predicted by Model:")
    print("    " + "-" * 50)
    print(f"    | {'Rank':<6} | {'Token':<18} | {'Probability':<15} |")
    print("    " + "-" * 50)
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        token_str = tokenizer.decode([idx.item()])
        print(f"    | #{rank:<5} | {repr(token_str):<18} | {prob.item() * 100:>6.2f}%         |")
    print("    " + "-" * 50)

    # 7. Greedy Choice (Argmax)
    best_token_id = top_indices[0].item()
    best_token_str = tokenizer.decode([best_token_id])
    print(f"\n[7] Greedy Decision (Argmax): {repr(best_token_str)}")
    print(f'    Next generated sentence preview: "{prompt}{best_token_str}"')
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    run_forward_pass_demo()
