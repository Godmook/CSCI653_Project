# CSCI653 — Scalable & Communication-Efficient Distributed Vision Training

A project that applies **hybrid parallelism** (DP × TP/PP) and **communication optimizations** (overlap, gradient compression, optimizer sharding) to efficiently train large-scale vision models (e.g., ViT) on multi-GPU/multi-node systems, and compares the approaches.

---

## (a) Problem Description
- In multi-node training, **all-reduce communication** and **memory usage** often become bottlenecks.
- Goal: **Reduce communication-bound time** and **improve scaling efficiency** while **preserving accuracy**.
- Scope: ViT-S/16 on CIFAR-10 → (resources permitting) ViT-B/16 on ImageNet-100/Imagenette.

---

## (b) Simulation Methods & Algorithms
- **Frameworks**: PyTorch Distributed (NCCL), FSDP / DeepSpeed ZeRO-2/3, Slurm/torchrun.
- **Parallelism**: Data Parallel (baseline) + optional Tensor/Pipeline Parallel.
- **Comm Opt**: gradient **bucketization & overlap**, choose **Ring vs Tree all-reduce**.
- **Compression**: PowerSGD or Top-k sparsification; 8-bit optimizer/grad if feasible.
- **Memory**: BF16/FP16, activation checkpointing, optimizer/parameter sharding.
- **I/O**: WebDataset shards, prefetch, pinned memory, local NVMe caching.

---

## (c) Expected Results
- **Strong/Weak scaling**: speedup & parallel efficiency curves.
- With overlap/compression, **lower communication time share (%)**.
- With FSDP/ZeRO, **reduced peak memory (GB)** → larger batch/models.
- **Minimal accuracy degradation** (small Δ top-1) and convergence speed comparison.
- Artifacts: `metrics.csv` + plots (throughput/efficiency/comm-share/accuracy).

---
