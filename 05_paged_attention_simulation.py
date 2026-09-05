"""
Milestone 5: PagedAttention & Block Tables (OS Paging for LLMs)
==============================================================
Concept:
In Milestone 4, we saw that the KV-cache grows linearly on every token.
Traditional inference servers allocated a contiguous block for max_len, wasting
60-80% of GPU RAM.

PagedAttention solves this using Operating Systems Virtual Memory Paging:
  1. Split KV-cache into fixed-size Physical Blocks (e.g. 4 tokens per block).
  2. Physical blocks can live anywhere in non-contiguous GPU RAM.
  3. A `BlockTable` maps each request's Logical Tokens -> Physical Blocks.
  4. Copy-on-Write (CoW): Parallel branches/agents share physical blocks with ref_counts!

This script is a self-contained, pure-Python simulation of vLLM's BlockManager.
"""

from typing import List, Dict, Optional

class PhysicalBlock:
    def __init__(self, block_id: int, block_size: int = 4):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0
        self.tokens: List[str] = []

    @property
    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - len(self.tokens)

    def append_token(self, token: str):
        if self.is_full:
            raise RuntimeError(f"Physical Block {self.block_id} is full!")
        self.tokens.append(token)

    def __repr__(self):
        return f"Block_{self.block_id}(refs={self.ref_count}, data={self.tokens})"


class BlockManager:
    """
    Manages GPU Virtual Memory page tables and physical block allocation.
    """
    def __init__(self, total_blocks: int = 10, block_size: int = 4):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.physical_blocks = [PhysicalBlock(i, block_size) for i in range(total_blocks)]
        self.free_list: List[int] = list(range(total_blocks)) # Free block IDs
        self.block_tables: Dict[str, List[int]] = {}          # request_id -> [block_ids]

    def allocate_block(self) -> int:
        if not self.free_list:
            raise MemoryError("❌ GPU Out of Memory (OOM)! No free physical blocks left.")
        block_id = self.free_list.pop(0)
        self.physical_blocks[block_id].ref_count = 1
        return block_id

    def free_request(self, request_id: str):
        """Reclaim memory when a request finishes"""
        if request_id not in self.block_tables:
            return
        for block_id in self.block_tables[request_id]:
            block = self.physical_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                block.tokens.clear()
                self.free_list.append(block_id)
        del self.block_tables[request_id]

    def append_token(self, request_id: str, token: str):
        """Append a token to a request, allocating new pages as needed"""
        if request_id not in self.block_tables:
            self.block_tables[request_id] = [self.allocate_block()]

        current_block_id = self.block_tables[request_id][-1]
        current_block = self.physical_blocks[current_block_id]

        # COPY-ON-WRITE CHECK:
        # If this block is shared by another branch (ref_count > 1), we cannot modify it!
        # We must allocate a fresh private block, copy the data, and write to it.
        if current_block.ref_count > 1 and not current_block.is_full:
            new_block_id = self.allocate_block()
            new_block = self.physical_blocks[new_block_id]
            new_block.tokens = list(current_block.tokens) # Copy data
            current_block.ref_count -= 1                 # Decrement shared ref
            self.block_tables[request_id][-1] = new_block_id
            current_block = new_block

        # If current block is full, allocate next page
        if current_block.is_full:
            new_block_id = self.allocate_block()
            self.block_tables[request_id].append(new_block_id)
            current_block = self.physical_blocks[new_block_id]

        current_block.append_token(token)

    def fork_request(self, parent_id: str, child_id: str):
        """
        Copy-on-Write (CoW) for parallel agent branching & beam search.
        Points the child request to the exact same physical blocks!
        """
        self.block_tables[child_id] = list(self.block_tables[parent_id])
        for block_id in self.block_tables[child_id]:
            self.physical_blocks[block_id].ref_count += 1

    def print_gpu_memory_state(self):
        print("\n" + "=" * 65)
        print(f" 🖥️  SIMULATED GPU VRAM STATE (Total Pages: {self.total_blocks} | Page Size: {self.block_size} tokens)")
        print("=" * 65)
        print(f"Free Blocks Queue: {self.free_list}")
        print("\nActive Page Tables (Logical -> Physical Mapping):")
        for req_id, blocks in self.block_tables.items():
            print(f"  • Request [{req_id:<12}]: Block IDs -> {blocks}")

        print("\nPhysical Blocks in GPU RAM:")
        for block in self.physical_blocks:
            status = f"IN USE (refs={block.ref_count})" if block.ref_count > 0 else "FREE"
            tokens_str = repr(block.tokens) if block.tokens else "[]"
            print(f"  [{block.block_id:02d}] {status:<18} | Content: {tokens_str}")
        print("=" * 65)


def run_paged_attention_demo():
    print("=" * 70)
    print(" 🚀 MILESTONE 5: PAGEDATTENTION & COPY-ON-WRITE SIMULATION")
    print("=" * 70)

    # Initialize manager with 8 physical blocks of 4 tokens each (Total 32 tokens capacity)
    manager = BlockManager(total_blocks=8, block_size=4)

    # 1. User A sends prompt (6 tokens)
    print("\n[Step 1] User A sends a 6-token prompt: ['def', ' quick', 'sort', '(', 'arr', ')']")
    for t in ['def', ' quick', 'sort', '(', 'arr', ')']:
        manager.append_token("User_A", t)
    manager.print_gpu_memory_state()

    # 2. Fork Request for Parallel Sampling (Agent Branching)
    print("\n[Step 2] FORK Request: Creating 'User_A_Branch2' to generate alternative code in parallel.")
    print("         -> Notice: NO extra memory is duplicated! Reference counts increment to 2.")
    manager.fork_request("User_A", "User_A_Branch2")
    manager.print_gpu_memory_state()

    # 3. Branch 1 and Branch 2 diverge (Copy-on-Write in action)
    print("\n[Step 3] Branch 1 generates: [':', '\\n', ' if']")
    for t in [':', '\n', ' if']:
        manager.append_token("User_A", t)

    print("\n[Step 4] Branch 2 generates alternative: [':', '\\n', ' pivot']")
    for t in [':', '\n', ' pivot']:
        manager.append_token("User_A_Branch2", t)
    manager.print_gpu_memory_state()

    # 4. User A finishes and disconnects
    print("\n[Step 5] User A finishes. Calling free_request('User_A')...")
    print("         -> Notice: Shared blocks are NOT deleted because Branch 2 is still reading them!")
    manager.free_request("User_A")
    manager.print_gpu_memory_state()

    # 5. Branch 2 finishes
    print("\n[Step 6] Branch 2 finishes. Calling free_request('User_A_Branch2')...")
    print("         -> Notice: All blocks successfully reclaimed to Free List!")
    manager.free_request("User_A_Branch2")
    manager.print_gpu_memory_state()

if __name__ == "__main__":
    run_paged_attention_demo()
