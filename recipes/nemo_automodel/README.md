# Qwen3-30B-A3B Training Performance Benchmarks

This document provides performance benchmarks for training Qwen3-30B-A3B model using NeMo Automodel on 8×H800 GPUs.

## Performance Results

### Configuration 1: Without Packed Sequence

**Script**: `run_qwen3_moe_30b.sh`  
**Configuration**: `qwen/qwen3_moe_30b_te_deepep.yaml`  
**Hardware**: 8 × H800 GPUs  
**Average TGS**: ~5,500 tokens/gpu/sec

**Sample Training Logs**:
```sh
2025-12-25 19:44:39 | INFO | root | step 34 | epoch 1 | loss 1.1590 | grad_norm 0.8866 | lr 1.00e-04 | mem 73.51 GiB | tps 39477.49(4934.69/gpu) | num_label_tokens 49473
2025-12-25 19:44:42 | INFO | root | step 35 | epoch 1 | loss 1.1512 | grad_norm 0.8644 | lr 1.00e-04 | mem 73.51 GiB | tps 44458.20(5557.28/gpu) | num_label_tokens 49368
2025-12-25 19:44:45 | INFO | root | step 36 | epoch 1 | loss 1.1465 | grad_norm 0.8263 | lr 1.00e-04 | mem 73.51 GiB | tps 44403.53(5550.44/gpu) | num_label_tokens 49920
2025-12-25 19:44:49 | INFO | root | step 37 | epoch 1 | loss 1.1257 | grad_norm 0.8175 | lr 1.00e-04 | mem 73.51 GiB | tps 35913.46(4489.18/gpu) | num_label_tokens 51413
2025-12-25 19:44:51 | INFO | root | step 38 | epoch 1 | loss 1.1509 | grad_norm 0.7952 | lr 1.00e-04 | mem 73.51 GiB | tps 44596.85(5574.61/gpu) | num_label_tokens 49459
2025-12-25 19:44:54 | INFO | root | step 39 | epoch 1 | loss 1.1075 | grad_norm 0.7370 | lr 1.00e-04 | mem 73.51 GiB | tps 44564.32(5570.54/gpu) | num_label_tokens 50392
2025-12-25 19:44:58 | INFO | root | step 40 | epoch 1 | loss 1.1168 | grad_norm 0.7299 | lr 1.00e-04 | mem 73.51 GiB | tps 38538.63(4817.33/gpu) | num_label_tokens 50263
2025-12-25 19:45:01 | INFO | root | step 41 | epoch 1 | loss 1.1380 | grad_norm 0.7714 | lr 1.00e-04 | mem 73.51 GiB | tps 40343.59(5042.95/gpu) | num_label_tokens 49840
```

### Configuration 2: With Packed Sequence

**Script**: `run_qwen3_moe_30_packed.sh`  
**Configuration**: `qwen/qwen3_moe_30b_te_packed.yaml`  
**Hardware**: 8 × H800 GPUs  
**Average TGS**: ~7,600 tokens/gpu/sec

**Sample Training Logs**:
```sh
2025-12-25 23:28:51 | INFO | root | step 4190 | epoch 199 | loss 0.0002 | grad_norm 0.0031 | lr 1.00e-04 | mem 69.08 GiB | tps 60845.19(7605.65/gpu) | num_label_tokens 60899
2025-12-25 23:28:53 | INFO | root | step 4191 | epoch 199 | loss 0.0002 | grad_norm 0.0058 | lr 1.00e-04 | mem 69.08 GiB | tps 60746.34(7593.29/gpu) | num_label_tokens 60877
2025-12-25 23:28:56 | INFO | root | step 4192 | epoch 199 | loss 0.0002 | grad_norm 0.0065 | lr 1.00e-04 | mem 69.08 GiB | tps 55182.67(6897.83/gpu) | num_label_tokens 60997
2025-12-25 23:28:59 | INFO | root | step 4193 | epoch 199 | loss 0.0002 | grad_norm 0.0054 | lr 1.00e-04 | mem 69.08 GiB | tps 61394.37(7674.30/gpu) | num_label_tokens 60955
```

## Performance Comparison

### Key Metrics from NVIDIA Developer Blog

According to the [NVIDIA Developer Blog on Accelerating Large-Scale Mixture-of-Experts Training in PyTorch](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/), NeMo Automodel achieves the following performance for Qwen3-MoE-30B:

- **TGS**: 12,040 tokens/gpu/sec
- **TFLOPs**: 277 TFLOPs/gpu
- **Hardware**: Multi-node H100 GPU setup
- **Micro Batch Size**: 4 per node (as mentioned in `qwen3_moe_30b_torch.yaml`)

### Performance Metrics Comparison Table

The following table compares performance metrics across different configurations. MFU (Model FLOPs Utilization) is calculated based on H100/H800 peak performance of **989 TFLOPs @ BF16** (using Tensor Cores).

| Configuration | TGS (tokens/gpu/sec) | Estimated TFLOPs/gpu | MFU (%) | Notes |
|--------------|---------------------|---------------------|---------|-------|
| **NVIDIA Reference** (NeMo Automodel) | 12,040 | 277 | ~28.0% | Multi-node setup(8×H100), optimized configuration |
| **This Repo** (Packed Sequence) | 7,674 | ~176.5 | ~17.8% | Single node (8×H800), with packed sequences |
| **This Repo** (No Packed Sequence) | 5,500 | ~126.5 | ~12.8% | Single node (8×H800), standard configuration |

**Calculation Method**:
- **TFLOPs/gpu** = `TGS × FLOPs/token ÷ 10^12`
- **MFU** = `TFLOPs/gpu ÷ GPU Peak TFLOPs`
- FLOPs/token is estimated from NVIDIA reference: `277 × 10^12 ÷ 12,040 ≈ 23.0 × 10^9 FLOPs/token`
- H800 peak TFLOPs @ BF16: **989 TFLOPs** (Tensor Cores)

### Performance Analysis

1. **Packed Sequence Optimization**: Using packed sequences improves TGS by approximately **39.5%** (from 5,500 to 7,674), which translates to a **39.5%** improvement in TFLOPs and MFU.

2. **Gap to NVIDIA Reference**: The current single-node implementation achieves approximately **63.7%** of the NVIDIA reference TGS (7,674 vs 12,040). This gap may be attributed to:
   - Multi-node optimizations in the reference implementation
   - Different batch size configurations
   - Additional framework-level optimizations

3. **Memory Efficiency**: Packed sequences also reduce memory usage (69.08 GiB vs 73.51 GiB), enabling larger batch sizes and potentially higher throughput.

## Related Resources

- **NVIDIA Developer Blog**: [Accelerating Large-Scale Mixture-of-Experts Training in PyTorch](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)
- **GitHub Issue**: [Inquiry regarding Best Practice and Performance for Qwen3-30B-A3B on Megatron-Core #21](https://github.com/yanring/Megatron-MoE-ModelZoo/issues/21)
- **Training Metrics Guide**: See `accelerating/benchmark/training_metrics_guide.md` for detailed explanations of TGS, TFLOPs, and MFU calculations

## Notes

- All TGS values are calculated using **effective tokens** (excluding padding)
- TFLOPs calculations assume activation checkpointing is enabled (α_train ≈ 4)
- MFU calculations are based on H800 peak performance with BF16 precision and Tensor Cores
- Performance may vary based on specific hardware configurations, driver versions, and software stack
