import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random

torch.manual_seed(42)

# 1. CONFIG

DIM = 128
TOTAL_FRAMES = 80

OCCLUSION_START = 25
OCCLUSION_END = 45

NOISE_VISIBLE = 0.03
NOISE_OCCLUDED = 0.12

MEMORY_SIZE = 20
TOPK = 5

# 2. MEMORY BANK

class MemoryBank:
    def __init__(self, size=20):
        self.bank = deque(maxlen=size)

    def add(self, feature):
        self.bank.append(feature.detach().clone())

    def retrieve(self, query, topk=5):
        """
        Retrieve most similar memories.
        """

        if len(self.bank) == 0:
            return query

        memories = torch.cat(list(self.bank), dim=0)

        # cosine similarity
        sim = F.cosine_similarity(query, memories)

        k = min(topk, len(memories))
        top_idx = torch.topk(sim, k=k).indices

        selected = memories[top_idx]

        # weighted average
        weights = F.softmax(sim[top_idx], dim=0)

        retrieved = (selected * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        return retrieved

# 3. GROUND TRUTH OBJECT

true_feature = F.normalize(torch.randn(1, DIM), dim=-1)

# Initial states
current_nomem = true_feature.clone()
current_mem = true_feature.clone()

memory = MemoryBank(size=MEMORY_SIZE)

# Warmup memory with first observations
memory.add(true_feature)

# 4. METRICS

sim_nomem_history = []
sim_mem_history = []

# 5. SIMULATION

for t in range(TOTAL_FRAMES):
    # Determine visibility
    occluded = OCCLUSION_START <= t <= OCCLUSION_END

    if occluded:
        noise_scale = NOISE_OCCLUDED
    else:
        noise_scale = NOISE_VISIBLE

    # Simulate noisy observation

    observation = true_feature + torch.randn(1, DIM) * noise_scale
    observation = F.normalize(observation, dim=-1)

    # BASELINE (NO MEMORY)

    current_nomem = observation

    # During occlusion:
    # model hallucinates / drifts
    if occluded:
        drift = torch.randn(1, DIM) * 0.08
        current_nomem = F.normalize(current_nomem + drift, dim=-1)

    # MEMORY BANK MODEL

    retrieved = memory.retrieve(observation, topk=TOPK)

    # Fuse observation with retrieved memory
    fused = 0.65 * observation + 0.35 * retrieved
    fused = F.normalize(fused, dim=-1)

    current_mem = fused

    # Update memory ONLY when visible
    if not occluded:
        memory.add(observation)

    # METRICS

    sim_nomem = F.cosine_similarity(
        current_nomem,
        true_feature
    ).item()

    sim_mem = F.cosine_similarity(
        current_mem,
        true_feature
    ).item()

    sim_nomem_history.append(sim_nomem)
    sim_mem_history.append(sim_mem)

# 6. VISUALIZATION

plt.figure(figsize=(12, 6))

plt.plot(sim_nomem_history,
         label='No Memory',
         linewidth=2)

plt.plot(sim_mem_history,
         label='Memory Bank',
         linewidth=2)

# Occlusion area
plt.axvspan(
    OCCLUSION_START,
    OCCLUSION_END,
    alpha=0.2,
    label='Occlusion Period'
)

plt.axhline(
    y=0.75,
    linestyle='--',
    label='Recognition Threshold'
)

plt.xlabel("Frame")
plt.ylabel("Cosine Similarity to Ground Truth")
plt.title("Object Identity Recovery After Occlusion")
plt.legend()
plt.grid(True)

plt.show()