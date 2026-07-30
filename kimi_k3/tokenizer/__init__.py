"""
kimi_k3/tokenizer/ — text tokenizer (deferred; roadmap Phase 2).

The architecture pass trains and tests on random token ids, so no tokenizer
exists yet. When training starts, this package will hold a BPE tokenizer
(train + load) targeting the kimi_1b_64k preset's 65,536-token vocab. See
docs/roadmap.md. This package exists so the slot is visible in the tree.
"""
