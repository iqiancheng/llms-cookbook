---
layout: post
title: "深度学习训练效率指标：TGS、TFLOPs 与 MFU 都是什么关系"
date: 2025-12-16
categories: [deep-learning, performance]
tags: [training, efficiency, metrics, gpu]
---

在大模型训练中，如何准确衡量系统效率？本文将系统性介绍三个核心指标：**TGS**（吞吐速度）、**TFLOPs**（算力占用）和 **MFU**（算力利用率），帮助你全面理解训练性能。

## TGS：衡量吞吐速度

**TGS (Tokens per GPU per Second)** 表示每块 GPU 每秒处理的有效 token 数，直接反映系统的吞吐能力。

**计算公式：**

$$
\text{TGS} = \frac{\text{gbs} \times \text{seqlen}}{\text{step\_time} \times \text{num\_gpus}}
$$

其中 **gbs** (Global Batch Size) = mbs × DP × grad_accum，包含以下参数：
- **mbs**: 每 GPU 的 micro batch size
- **DP**: 数据并行度
- **grad_accum**: 梯度累积步数
- **seqlen**: 每样本的有效序列长度
- **step_time**: 单步训练耗时（秒）
- **num_gpus**: 参与训练的 GPU 总数

TGS 主要受以下因素影响：**并行通信效率**（梯度同步、参数传输是否高效，通信与计算是否重叠）、**批大小可达性**（显存限制下能否维持足够的 batch size 和序列长度）、**算子优化**（Flash Attention、fused kernels、CUDA Graph 等）、**数据管线**（数据加载效率、padding 策略）以及**负载均衡**（MoE 路由均衡性、I/O 稳定性）。

**关于 Padding：** TGS 应按有效 token 计算，不包含 padding。如果算子能跳过 padding（如 Flash Attention 的 packed sequences），这些 pad token 不应计入分子，否则会导致指标虚高、失去可比性。若框架无法统计有效长度，只能用 `gbs × max_seq_len`，需明确标注"含 padding"。

## FLOPs/token：衡量计算密度

**FLOPs/token** 表示处理每个 token 所需的浮点运算量，反映模型的算力密度。

**计算公式：**

$$
\text{FLOPs/token} = L \cdot \left( F_{\text{attn}} + F_{\text{mlp}} \cdot k_{\text{MoE}} + F_{\text{others}} \right) \cdot \alpha_{\text{train}}
$$

参数说明：
- **L**: 层数
- **F_attn**: 注意力层的 FLOPs，依赖于 `d_model`、`num_heads`、`d_head`、`seqlen`
- **F_mlp**: MLP 层的 FLOPs，依赖于 `d_model`、`d_ff`（前馈维度）
- **k_MoE**: 每 token 激活的专家数（仅对 MoE 模型，如 Qwen3-30B-A3B 中 k=8）
- **α_train**: 训练系数，包含前向、反向和激活重计算的开销
  - 无重计算：α ≈ 2-3
  - 启用重计算：α ≈ 4

FLOPs/token 主要由**模型架构**（层数、隐藏维度、头数、前馈维度）、**MoE 配置**（激活专家数 k，仅影响 MLP 部分）、**序列长度**（注意力的计算复杂度与 seqlen 相关）以及**训练策略**（是否启用激活重计算）决定。

## TFLOPs：从吞吐到算力

**TFLOPs/gpu** 表示每块 GPU 每秒执行的万亿次浮点运算，综合反映吞吐速度和计算密度。

**计算公式：**

$$
\text{TFLOPs/gpu} = \frac{\text{TGS} \times \text{FLOPs/token}}{10^{12}}
$$

这个公式建立了吞吐与算力的桥梁：**TGS** 决定"跑得有多快"，**FLOPs/token** 决定"每步有多重"，**TFLOPs** 反映"算力吃了多少"。

在相同 TGS 下，模型越大、k 越高、序列越长，FLOPs/token 越大，TFLOPs 越高。但在相同硬件上，FLOPs/token 越大会导致 step_time 越长，从而降低 TGS。因此，TGS 和 TFLOPs 相互制约，共同刻画训练效率。

## MFU：算力利用率

**MFU (Model FLOPs Utilization)** 表示实际算力占 GPU 理论峰值的比例。

**计算公式：**

$$
\text{MFU} = \frac{\text{TFLOPs/gpu}}{\text{GPU 峰值 TFLOPs}}
$$

MFU 接近 1 表示 GPU 算力被充分利用。实际训练中，MFU 通常在 0.3-0.6 之间，受内存带宽、算子效率、并行开销等因素限制。

## 三个指标的协同价值

这三个指标从不同维度反映训练效率：

| 指标 | 反映维度 | 关键问题 |
|------|---------|---------|
| **TGS** | 吞吐速度 | 系统每秒能推进多少训练？|
| **TFLOPs** | 算力占用 | GPU 每秒执行了多少计算？|
| **MFU** | 算力利用率 | GPU 潜力被发挥了多少？|

通过联合分析，可以准确定位性能瓶颈：

**TGS 低 + FLOPs/token 高** 说明瓶颈在通信、调度、I/O 或内核效率不足，模型算力密度高但系统跑不动。

**TGS 高 + FLOPs/token 低** 说明模型较"轻"，算力利用率不高，优化方向是增大 batch size、减少 padding、优化算子。

**MFU 低** 说明瓶颈在内存带宽、算子选择、混合精度策略，需要底层优化（kernel fusion、张量核利用）。

## 最佳实践

在报告训练性能时，建议同时提供以下四项指标：

**1. TGS**（tokens/gpu/sec）- 明确是否含 padding，说明 grad_accum 和并行配置

**2. FLOPs/token** - 注明是否含前向+反向+重计算，对 MoE 模型说明激活专家数 k

**3. TFLOPs/gpu** - 用公式 `TGS × FLOPs/token ÷ 1e12` 计算

**4. MFU** - 说明 GPU 型号和峰值算力，明确精度（FP16/BF16/FP8）

示例报告格式：

```
模型：Qwen3-30B-A3B (L=48, k=8)
硬件：8×A100 80GB
配置：DP=4, TP=2, seqlen=4096, mbs=2, grad_accum=4

TGS: 1250 tokens/gpu/sec (有效 token，不含 padding)
FLOPs/token: 2.8×10^12 (含前向+反向+重计算, α=4)
TFLOPs/gpu: 3.5 TFLOPs
MFU: 11.2% (A100 峰值 312 TFLOPs @ BF16)
```

## 总结

- **TGS** 看系统吞吐与并行效率："跑得快不快"
- **FLOPs/token** 看每 token 的算力强度："每步有多重"
- **TFLOPs** 看算力吞吐："算力吃了多少"
- **MFU** 看算力利用率："吃得饱不饱"

单一指标无法全面反映训练效率，只有联合分析才能准确定位瓶颈、指导优化。在实际工作中，从 TGS 入手评估吞吐，通过 TFLOPs/MFU 评估算力利用，最终实现"又快又满"的高效训练。

---

## 参考资源

1. **Megatron-LM FLOPs Calculation** - Paper: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (SC '21) | GitHub: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - 详细的 FLOPs 计算公式和 MFU 测量方法，支持大规模并行训练

2. **DeepSpeed Performance Profiling** - Flops Profiler: [DeepSpeed Flops Profiler Tutorial](https://www.deepspeed.ai/tutorials/flops-profiler/) | GitHub: [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) - 提供模块级性能分析、FLOPs 计数和瓶颈识别工具

3. **Scaling Laws for Neural Language Models** - Paper: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) - 首次系统性研究模型规模、数据量和计算量之间的幂律关系，指导了 GPT-3 等大模型的训练策略

4. **Training Compute-Optimal Large Language Models (Chinchilla)** - Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) - 提出模型参数与训练 tokens 应等比例扩展（每翻倍参数，tokens 也应翻倍），Chinchilla (70B) 以更小规模超越 Gopher (280B) 和 GPT-3 (175B)

5. **PaLM: Scaling Language Modeling with Pathways** - Paper: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (Chowdhery et al., 2022) | Blog: [Google Research Blog](https://research.google/blog/pathways-language-model-palm-scaling-to-540-billion-parameters-for-breakthrough-performance/) - 540B 参数模型在 6144 TPU v4 上达到 57.8% 硬件 FLOPs 利用率，展示了 Pathways 系统的高效训练能力和 scaling 收益