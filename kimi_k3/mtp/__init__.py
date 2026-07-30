"""
kimi_k3/mtp/ — Multi-Token Prediction heads (deferred / optional).

Kimi K2/K3 attach extra "next-n-predict" layers (config alias
`num_nextn_predict_layers`) that predict several future tokens per step —
useful for training signal and speculative decoding. It is faithful to K3
but not required for architecture study, and it only pays off once the
training loop exists (roadmap Phase 2+), so it is intentionally deferred.
This package exists so the slot is visible in the tree.
"""
