"""
kimi_k3/training/ — training loop (deferred; roadmap Phase 2+).

Will hold the loss (next-token cross-entropy, plus MTP heads if added), the
optimizer setup, checkpointing, and the progressive-length training schedule
for 64K → 256K. Nothing is implemented during the architecture pass — the
current codebase is a verified forward/decode reference only. See
docs/roadmap.md. This package exists so the slot is visible in the tree.
"""
