"""
Milestone 1: The Tokenizer Demystified
======================================
Concept:
Computers cannot read characters or words directly. Neural networks only do
matrix math on numbers. 

The Tokenizer is the translation layer:
  Text ("Hello world") <---> Token IDs ([9906, 1917]) <---> High-Dimensional Vectors

In this script, you will learn:
1. How text is broken down into sub-words via Byte-Pair Encoding (BPE).
2. How to inspect the vocabulary dictionary.
3. How whitespace, punctuation, and special control tokens (<|im_start|>, <|im_end|>) work.
"""

from transformers import AutoTokenizer

# We use SmolLM-135M: a modern, open-source ultra-compact LLM by HuggingFace
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

def run_tokenizer_demo():
    print("=" * 70)
    print(" 🚀 MILESTONE 1: THE TOKENIZER UNDER THE HOOD")
    print("=" * 70)

    print(f"\n[1] Loading tokenizer for: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 1. Vocabulary stats
    vocab_size = len(tokenizer)
    print(f"    ✓ Vocabulary Size: {vocab_size:,} unique tokens")
    print(f"    ✓ BOS (Beginning of Stream) Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"    ✓ EOS (End of Stream) Token:       {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # 2. Tokenizing a sample sentence
    sample_text = "Distributed systems power modern AI infrastructure."
    print(f"\n[2] Tokenizing sample sentence:")
    print(f'    Raw Text: "{sample_text}"')
    
    # Text -> Token IDs
    token_ids = tokenizer.encode(sample_text)
    print(f"    Token IDs (Integers): {token_ids}")

    # 3. Sub-word breakdown inspection
    print("\n[3] Inspecting Token-by-Token Decomposition:")
    print("    " + "-" * 55)
    print(f"    | {'Index':<6} | {'Token ID':<10} | {'Subword / Text Piece':<25} |")
    print("    " + "-" * 55)
    for idx, token_id in enumerate(token_ids):
        # Decode individual token back to string
        piece = tokenizer.decode([token_id])
        # Make whitespace visible for educational clarity
        display_piece = repr(piece)
        print(f"    | {idx:<6} | {token_id:<10} | {display_piece:<25} |")
    print("    " + "-" * 55)

    # 4. Chat Templates & Special Tokens (The plumbing of AI agents)
    print("\n[4] Chat Templates (How System & User Prompts are structured):")
    messages = [
        {"role": "system", "content": "You are a distributed systems expert."},
        {"role": "user", "content": "Explain KV-cache in 1 line."}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("    Formatted Raw Prompt with Control Tokens:")
    for line in formatted_prompt.strip().split("\n"):
        print(f"    │ {line}")

    # 5. Roundtrip verification
    decoded_text = tokenizer.decode(token_ids)
    print(f"\n[5] Decoding IDs back to text: '{decoded_text}'")
    assert decoded_text == sample_text, "Roundtrip decoding mismatch!"
    print("    ✓ Roundtrip Assertion Passed: Text -> IDs -> Text is lossless!")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    run_tokenizer_demo()
