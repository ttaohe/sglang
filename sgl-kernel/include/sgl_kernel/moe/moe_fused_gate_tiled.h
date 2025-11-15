// Copyright 2025 SGLang
//
// Tiled MoE fused gate public declarations.
// This header is installed under the project include/ tree to follow common C++ layout conventions.
// Source implementations reside in csrc/moe/.
#pragma once

#include <torch/all.h>
#include <vector>

// Tiled moe fused gate dynamic dispatcher (VPT > 32 fallback).
// Launches a dynamic tiled CUDA kernel specialized by dtype internally.
std::vector<at::Tensor> moe_fused_gate_tiled(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

// Tiled moe fused gate static specializations (compile-time params inside).
// Launches static tiled CUDA kernels for common shapes (e.g., 384/64 experts with group=1).
std::vector<at::Tensor> moe_fused_gate_tiled_static(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);

// Keep a consistent launch style with LAUNCH_MOE_GATE_CONFIG.
// Note: dtype dispatch (bf16/fp16/fp32) is performed inside moe_fused_gate_tiled_static,
// so callers do not need to branch on dtype when using this macro.
#define LAUNCH_MOE_GATE_TILED_CONFIG(EXPERTS, EXPERT_GROUP, TILE)                                        \
  do {                                                                                                   \
    (void)(EXPERTS);                                                                                     \
    (void)(EXPERT_GROUP);                                                                                \
    (void)(TILE);                                                                                        \
    return moe_fused_gate_tiled_static(                                                                  \
        input,                                                                                           \
        bias,                                                                                            \
        num_expert_group,                                                                                \
        topk_group,                                                                                      \
        topk,                                                                                            \
        num_fused_shared_experts,                                                                        \
        routed_scaling_factor,                                                                           \
        apply_routed_scaling_factor_on_output);                                                          \
  } while (0)


