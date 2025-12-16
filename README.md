# LLM Training Cookbook 🍳

Welcome to the LLM Training Cookbook! This repository serves as a collection of recipes, acceleration techniques, and best practices for training Large Language Models (LLMs) effectively and efficiently.

## 📚 Introduction

Training large models requires a deep understanding of distributed systems, memory management, and optimization techniques. This cookbook aims to demystify these concepts and provide actionable guides for researchers and engineers.

## 🍲 Recipes

### Frameworks & Tools
- **Megatron-LM**: Best for pre-training massive models from scratch.
- **DeepSpeed**: Excellent for fine-tuning and mixed-precision training.
- **FSDP (Fully Sharded Data Parallel)**: PyTorch native solution for distributed training.
- **Hugging Face Trainer**: High-level API for quick experimentation.

### Model Specifics
- **Qwen2 / Qwen3**: Tips for handling Qwen's specific architecture (e.g., dynamic ntk).
- **Qwen3-vl-moe**: Handling MoE (Mixture of Experts) models efficiently.

## 🚀 Acceleration Techniques

### Memory Optimization
- **Gradient Checkpointing**: Trade compute for memory by recomputing activations.
- **Flash Attention 2 / 3**: Faster attention mechanisms that scale linearly with sequence length.
- **Mixed Precision (BF16/FP16)**: Reduce memory usage and increase throughput.
- **Quantization (QLoRA, AWQ)**: Train larger models on consumer hardware.

### Distributed Training Strategies
- **Data Parallel (DP / DDP)**: Replicate model across GPUs.
- **Tensor Parallel (TP)**: Split individual layers across GPUs (intra-layer).
- **Pipeline Parallel (PP)**: Split model layers across GPUs (inter-layer).
- **Context Parallel (CP)**: Split sequence dimension for ultra-long context training.

## 🛠️ Best Practices

### Environment Setup
- **Docker**: Always use containerized environments for reproducibility.
- **Conda/Mamba**: Manage python dependencies strictly.
- **Flash Attention Installation**: Often tricky; compiled from source is recommended for performance.

### Monitoring & Debugging
- **WandB / MLflow**: Vital for tracking loss curves and system metrics.
- **TensorBoard**: Good for visualizing graph execution.
- **NCCL Tests**: Verify inter-GPU communication bandwidth before training.
- **OOM Troubleshooting**:
    1. Check `max_split_size_mb`.
    2. Reduce batch size.
    3. Enable CPU offloading (if acceptable performance hit).

## 🔗 Resources

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/docs/config-json/)
- [Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM)
- [Flash Attention Repository](https://github.com/Dao-AILab/flash-attention)

---

*Contributions are welcome! Please open a PR to add your favorite recipe.*
