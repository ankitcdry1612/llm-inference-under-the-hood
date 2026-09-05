# ⚡ LLM Inference Under The Hood
> **A first-principles, code-first guide for backend & distributed systems engineers to understand how Large Language Models actually run and scale on hardware.**

---

## 🎯 Why This Repo Exists

Most AI tutorials fall into two extremes:
1. **Academic ML Math:** Complex mathematical proofs and differential calculus.
2. **API Glue Code:** Superficial `openai.chat.completions.create()` wrappers.

Neither explains how an LLM **actually executes on hardware**:
* *Why are LLM inference servers memory-bandwidth bound instead of compute-bound?*
* *What does the KV-Cache actually look like in RAM?*
* *How did 1970s OS Virtual Memory Paging solve GPU memory fragmentation (PagedAttention)?*
* *How do inference engines schedule requests at the token-iteration level (Continuous Batching)?*

This repository breaks open the black box. Every milestone is a self-contained, heavily-commented Python script designed to be run on your laptop in 10 seconds.

---

## 🗺️ The 6 Interactive Milestones

| Milestone | Script | What You Will Learn |
| :--- | :--- | :--- |
| **01** | [`01_the_tokenizer.py`](01_the_tokenizer.py) | How text translates to integer Token IDs, Byte-Pair Encoding (BPE), vocabulary tables, and chat templates. |
| **02** | [`02_the_forward_pass_and_logits.py`](02_the_forward_pass_and_logits.py) | Passing tensors through Transformer layers, inspecting raw **Logits** across 49k vocabulary words, and Softmax probabilities. |
| **03** | [`03_manual_autoregressive_loop.py`](03_manual_autoregressive_loop.py) | Writing the token generation `while` loop completely from scratch (no `.generate()`), implementing temperature sampling, and discovering the $O(N^2)$ compute flaw. |
| **04** | [`04_the_kv_cache_deep_dive.py`](04_the_kv_cache_deep_dive.py) | Inspecting the `past_key_values` tensor, measuring KV-cache memory growth per token, and passing $[1, 1]$ token inputs to avoid recomputing past tokens. |
| **05** | [`05_paged_attention_simulation.py`](05_paged_attention_simulation.py) | A pure-Python simulation of **PagedAttention Block Tables** and **Copy-on-Write (CoW)** for parallel agent branching and beam search. |
| **06** | [`06_continuous_batching_scheduler.py`](06_continuous_batching_scheduler.py) | An **Iteration-Level Scheduler** simulation demonstrating real-time request eviction and admission to eliminate padding `<pad>` waste. |

---

## 🚀 Quickstart (Run on Mac / Linux / Windows)

No NVIDIA GPU required! All milestones use **SmolLM-135M** (a modern, ultra-compact open-weights LLM by HuggingFace) that runs blisteringly fast on any CPU or Apple Silicon.

```bash
# 1. Clone the repository
git clone https://github.com/ankitcdry1612/llm-inference-under-the-hood.git
cd llm-inference-under-the-hood

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install lightweight dependencies (PyTorch + Transformers)
pip install -r requirements.txt

# 4. Run any milestone!
python 01_the_tokenizer.py
python 02_the_forward_pass_and_logits.py
python 03_manual_autoregressive_loop.py
python 04_the_kv_cache_deep_dive.py
python 05_paged_attention_simulation.py
python 06_continuous_batching_scheduler.py
```

---

## 🏗️ The Core Architecture Mental Model

```
Raw Text Prompt
       │
       ▼
[01: Tokenizer] ─────────▶ Token IDs [16535, 4663, 1734...]
                                │
                                ▼
[02: Forward Pass] ──────▶ Transformer Layers (Attention + FFN)
                                │
                                ▼
                           Raw Logits Vector [49,152 vocab scores]
                                │
                                ▼
[03: Sampling / Argmax] ─▶ Select Next Token ID
                                │
       ┌────────────────────────┴────────────────────────┐
       ▼                                                 ▼
[04: Update KV-Cache]                         [06: Continuous Batching]
Cache Keys & Values in RAM                     Schedule on ~20ms clock ticks
       │                                                 │
       ▼                                                 ▼
[05: PagedAttention]                          Evict on <EOS>, Inject new requests
Map Logical Pages -> Physical Blocks           Zero <pad> waste
```

---

## 👤 Author

**Ankit Chaudhary**  
*Distributed Systems Engineer @ Microsoft (Azure Messaging) | ex-Disney+ Hotstar | IIIT Delhi*  
* [X (Twitter): @ankitcdry](https://x.com/ankitcdry)  
* [LinkedIn: in/ankit17022](https://www.linkedin.com/in/ankit17022/)
