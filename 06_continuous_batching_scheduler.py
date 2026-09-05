"""
Milestone 6: Continuous (Iteration-Level) Batching Scheduler
============================================================
Concept:
In traditional web servers, batching is done at the Request level:
  Wait for N requests -> Process together -> Return responses together.

On GPUs, this is disastrous: a 5-token request gets blocked for 30 seconds
waiting for a 1,000-token request (wasting compute on <pad> dummy tokens).

Continuous Batching solves this:
  1. The server runs clock ticks (iterations) every ~20ms.
  2. On EVERY clock tick, finished requests (<EOS>) are evicted immediately.
  3. Waiting requests from the queue are injected into freed slots for the next tick!

This script is a pure Python simulation of an Iteration-Level Scheduler.
"""

import time
from typing import List, Optional

class InferenceRequest:
    def __init__(self, request_id: str, prompt: str, total_tokens_needed: int, arrival_tick: int):
        self.request_id = request_id
        self.prompt = prompt
        self.total_tokens_needed = total_tokens_needed # How many tokens to generate
        self.tokens_generated = 0
        self.arrival_tick = arrival_tick
        self.completed_tick: Optional[int] = None

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.total_tokens_needed

    def step(self):
        """Simulate generating 1 token in this iteration"""
        self.tokens_generated += 1


class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int = 3):
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[InferenceRequest] = []
        self.running_batch: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []
        self.current_tick = 0

    def add_request(self, req: InferenceRequest):
        self.waiting_queue.append(req)

    def step_iteration(self):
        """Execute one ~20ms GPU Clock Tick"""
        self.current_tick += 1
        print(f"\n⏰ [TICK {self.current_tick:02d}] ------------------------------------------")

        # 1. EVICTION: Remove finished requests immediately!
        new_running = []
        for req in self.running_batch:
            if req.is_finished:
                req.completed_tick = self.current_tick - 1
                self.completed_requests.append(req)
                print(f"  🟢 EVICTED: [{req.request_id}] Finished in {req.total_tokens_needed} tokens! Sent response to client.")
            else:
                new_running.append(req)
        self.running_batch = new_running

        # 2. ADMISSION / INJECTION: Fill empty slots from waiting queue
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            # Check if request has arrived by this tick
            if self.waiting_queue[0].arrival_tick <= self.current_tick:
                new_req = self.waiting_queue.pop(0)
                self.running_batch.append(new_req)
                print(f"  📥 INJECTED: [{new_req.request_id}] (Needs {new_req.total_tokens_needed} tokens) into batch slot #{len(self.running_batch)}")
            else:
                break

        # 3. GPU FORWARD PASS: Generate 1 token for all active requests in parallel
        if not self.running_batch:
            print("  💤 GPU Idle (Queue empty)")
            return False

        print(f"  ⚡ GPU Running Forward Pass on {len(self.running_batch)} Active Slots:")
        for idx, req in enumerate(self.running_batch, start=1):
            req.step()
            bar = "█" * req.tokens_generated + "░" * (req.total_tokens_needed - req.tokens_generated)
            print(f"     Slot {idx}: [{req.request_id:<10}] |{bar}| ({req.tokens_generated}/{req.total_tokens_needed} tokens)")

        return True


def run_continuous_batching_demo():
    print("=" * 70)
    print(" 🚀 MILESTONE 6: CONTINUOUS (ITERATION-LEVEL) SCHEDULER SIMULATION")
    print("=" * 70)
    print("Scenario: Batch Size = 2 Slots on GPU.")
    print("• User A arrives at Tick 1 (Needs 12 tokens - long query)")
    print("• User B arrives at Tick 1 (Needs 3 tokens - short query)")
    print("• User C arrives at Tick 2 (Needs 4 tokens - queued query)")
    print("-" * 70)

    scheduler = ContinuousBatchScheduler(max_batch_size=2)
    
    # Add requests
    scheduler.add_request(InferenceRequest("User_A (Long)",  "Write essay",      total_tokens_needed=10, arrival_tick=1))
    scheduler.add_request(InferenceRequest("User_B (Short)", "What is 2+2?",     total_tokens_needed=3,  arrival_tick=1))
    scheduler.add_request(InferenceRequest("User_C (Med)",   "Summarize news",   total_tokens_needed=4,  arrival_tick=2))

    # Run scheduler until all requests are done
    while True:
        active = scheduler.step_iteration()
        # If no active requests and waiting queue is empty, we are done
        if not active and not scheduler.waiting_queue:
            break

    print("\n" + "=" * 70)
    print(" 📊 SCHEDULING BENCHMARK REPORT")
    print("=" * 70)
    print(f"| {'Request ID':<16} | {'Tokens':<8} | {'Arrival':<10} | {'Completed':<12} | {'Turnaround Time':<15} |")
    print("-" * 70)
    for req in scheduler.completed_requests:
        turnaround = req.completed_tick - req.arrival_tick + 1
        print(f"| {req.request_id:<16} | {req.total_tokens_needed:<8} | Tick {req.arrival_tick:<5} | Tick {req.completed_tick:<7} | {turnaround} ticks       |")
    print("-" * 70)
    print("\n💡 Key Observation:")
    print("   User_B finished in 3 ticks and was IMMEDIATELY returned to the user.")
    print("   User_C was admitted on Tick 4 without waiting for User_A (who took 10 ticks).")
    print("   In Static Batching, User_B would have waited all 10 ticks with 7 dummy <pad> tokens!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_continuous_batching_demo()
