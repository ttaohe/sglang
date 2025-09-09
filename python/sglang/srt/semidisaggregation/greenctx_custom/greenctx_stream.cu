#include <torch/all.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cuda_utils.h"
#include <torch/extension.h>
#include "greenctx_stream.h"

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  CUstream streamA;
  CUstream streamB;

  unsigned int nbGroups = 1;

  if (smA <= 0 || smB <= 0) {
    TORCH_CHECK(false, "SM counts must be positive");
  }

  // Initialize device
  CUDA_RT(cudaInitDevice(device, 0, 0));

  // Query input SMs
  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input, CU_DEV_RESOURCE_TYPE_SM));
  // We want 3/4 the device for our green context
  unsigned int minCount = (unsigned int)(smA + smB);
  unsigned int minCountA = (unsigned int)(smA);

  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  // Split resources
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));

  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));

  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device, CU_GREEN_CTX_DEFAULT_STREAM));

  CUDA_DRV(cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  int smCountA = resources[0].sm.smCount;
  int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> vec = {(int64_t)streamA, (int64_t)streamB, smCountA, smCountB};
  return vec;
}



// 包含SM分配约束的结构体
struct SMConstraints {
    int min_count;
    int alignment;
};

int get_sm_available(int device) {
    cudaDeviceProp props;
    CUDA_RT(cudaGetDeviceProperties(&props, device));
    return props.multiProcessorCount;
}

std::pair<int, int> get_compute_capability(int device) {
    cudaDeviceProp props;
    CUDA_RT(cudaGetDeviceProperties(&props, device));
    return {props.major, props.minor};
}

SMConstraints get_sm_constraints(int device) {
    auto [major, minor] = get_compute_capability(device);
    if (major == 6) return {1, 1}; // Pascal
    if (major == 7) return {2, 2}; // Volta/Turing
    if (major == 8) return {4, 2}; // Ampere
    if (major >= 9) return {8, 8}; // Hopper+
    return {1, 1}; // Fallback for unknown arch
}

std::pair<int, int> find_optimal_sm_split(
    double target_percent_a,
    double target_percent_b,
    int device
) {
    int total_sms = get_sm_available(device);
    SMConstraints constraints = get_sm_constraints(device);
    int min_count = constraints.min_count;
    int alignment = constraints.alignment;

    double target_sm_a = target_percent_a * total_sms;
    double target_sm_b = target_percent_b * total_sms;

    int best_sm_a = -1, best_sm_b = -1;
    double best_error = std::numeric_limits<double>::max();

    int max_units_a = total_sms / alignment;

    // 遍历所有可能的、符合对齐要求的 sm_a 分配
    for (int units_a = 0; units_a <= max_units_a; ++units_a) {
        int sm_a = units_a * alignment;

        // 跳过不满足最小要求的分配 (除非为0)
        if (sm_a > 0 && sm_a < min_count) {
            continue;
        }

        // 计算剩余的SMs
        int remaining_sms = total_sms - sm_a;
        int sm_b = 0;

        // 如果剩余SMs足够，则为sm_b分配
        if (remaining_sms >= min_count) {
            int units_b = remaining_sms / alignment;
            sm_b = units_b * alignment;
            // 再次检查，确保对齐后的sm_b仍满足最小要求
            if (sm_b < min_count) {
                sm_b = 0;
            }
        }

        // 跳过无效的分配
        if (sm_a == 0 && sm_b == 0) {
            continue;
        }
        if (sm_a + sm_b > total_sms) {
            continue;
        }

        // 计算误差（目标SM数与实际SM数的欧几里得距离）
        double error_a = static_cast<double>(sm_a) - target_sm_a;
        double error_b = static_cast<double>(sm_b) - target_sm_b;
        double total_error = std::sqrt(error_a * error_a + error_b * error_b);

        // 如果找到更好的方案，则更新
        if (total_error < best_error) {
            best_error = total_error;
            best_sm_a = sm_a;
            best_sm_b = sm_b;
        }
    }

    if (best_sm_a == -1) {
        throw std::runtime_error(
            "Failed to find a valid SM split satisfying architecture constraints. Total SMs: " +
            std::to_string(total_sms) + ", Min Count: " + std::to_string(min_count) +
            ", Alignment: " + std::to_string(alignment));
    }
    
    return {best_sm_a, best_sm_b};
}

/**
 * @brief 通过百分比创建 greenctx 流，自动找到最接近的 SM 分配方案
 * 
 * @param target_percent_a 流 A 的目标 SM 百分比 (0.0 到 1.0)
 * @param target_percent_b 流 B 的目标 SM 百分比 (0.0 到 1.0)
 * @param device GPU 设备 ID
 * @param verbose 是否打印详细的分配信息
 * @return SMAllocationResult 包含 SM 分配、实际百分比和流指针的结果
 */
SMAllocationResult create_greenctx_stream_by_percent(
    float target_percent_a,
    float target_percent_b,
    int device
) {
    // 1. 输入验证
    TORCH_CHECK(target_percent_a >= 0.0 && target_percent_a <= 1.0 &&
                target_percent_b >= 0.0 && target_percent_b <= 1.0,
                "Percentages must be between 0.0 and 1.0.");

    TORCH_CHECK(std::abs(target_percent_a + target_percent_b - 1.0) < 1e-6,
                "Sum of percentages must be 1.0. Current sum: " +
                std::to_string(target_percent_a + target_percent_b));
    
    // 如果一个百分比是100%，另一个是0%，则特殊处理
    if (target_percent_a == 1.0 || target_percent_b == 1.0) {
        int total_sms = get_sm_available(device);
        SMConstraints constraints = get_sm_constraints(device);
        int aligned_total = (total_sms / constraints.alignment) * constraints.alignment;
        
        int64_t sm_a = (target_percent_a == 1.0) ? aligned_total : 0;
        int64_t sm_b = (target_percent_b == 1.0) ? aligned_total : 0;
        
        // if (verbose) {
        //     std::cout << "Handling 100% allocation case. SMs: " << sm_a << " / " << sm_b << std::endl;
        // }
        
        std::vector<int64_t> res = create_greenctx_stream_by_value(sm_a, sm_b, device);
        
        SMAllocationResult result;
        result.streamA_ptr = res[0];
        result.streamB_ptr = res[1];
        result.sm_a = res[2];
        result.sm_b = res[3];
        result.actual_percent_a = (total_sms > 0) ? static_cast<double>(result.sm_a) / total_sms : 0.0;
        result.actual_percent_b = (total_sms > 0) ? static_cast<double>(result.sm_b) / total_sms : 0.0;
        return result;
    }


    auto [sm_a, sm_b] = find_optimal_sm_split(target_percent_a, target_percent_b, device);

    std::vector<int64_t> res = create_greenctx_stream_by_value(sm_a, sm_b, device);

    SMAllocationResult result;
    result.streamA_ptr = res[0];
    result.streamB_ptr = res[1];
    
    // 使用从`by_value`返回的真实SM数量，这是最准确的
    result.sm_a = res[2]; 
    result.sm_b = res[3]; 

    // 检查实际分配是否与请求的匹配
    TORCH_CHECK(result.sm_a == sm_a && result.sm_b == sm_b,
                "Mismatch between requested and allocated SMs. Requested: " +
                std::to_string(sm_a) + "/" + std::to_string(sm_b) +
                ", Got: " + std::to_string(result.sm_a) + "/" + std::to_string(result.sm_b));

    int total_sms_final = get_sm_available(device);
    result.actual_percent_a = (total_sms_final > 0) ? static_cast<double>(result.sm_a) / total_sms_final : 0.0;
    result.actual_percent_b = (total_sms_final > 0) ? static_cast<double>(result.sm_b) / total_sms_final : 0.0;
    
    return result;
}


// 绑定到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SMAllocationResult>(m, "SMAllocationResult")
        .def(py::init<>()) // 绑定默认构造函数
        // 绑定所有成员变量，让它们可以在 Python 中被访问
        .def_readwrite("streamA_ptr", &SMAllocationResult::streamA_ptr)
        .def_readwrite("streamB_ptr", &SMAllocationResult::streamB_ptr)
        .def_readwrite("sm_a", &SMAllocationResult::sm_a)
        .def_readwrite("sm_b", &SMAllocationResult::sm_b)
        .def_readwrite("actual_percent_a", &SMAllocationResult::actual_percent_a)
        .def_readwrite("actual_percent_b", &SMAllocationResult::actual_percent_b)
        // (可选但推荐) 添加一个 __repr__ 方法，方便在 Python 中打印调试
        .def("__repr__",
             [](const SMAllocationResult &a) {
                 std::ostringstream ss;
                 ss << "<SMAllocationResult: "
                    << "sm_a=" << a.sm_a << ", sm_b=" << a.sm_b
                    << ", actual_percent_a=" << std::fixed << std::setprecision(3) << a.actual_percent_a
                    << ", actual_percent_b=" << std::fixed << std::setprecision(3) << a.actual_percent_b
                    << ", streamA_ptr=0x" << std::hex << a.streamA_ptr
                    << ", streamB_ptr=0x" << std::hex << a.streamB_ptr
                    << ">";
                 return ss.str();
             });
    m.def("create_greenctx_stream_by_percent", &create_greenctx_stream_by_percent,
          "Create GreenCtx streams by percentage");
    m.def("create_greenctx_stream_by_value", &create_greenctx_stream_by_value,
          "Create GreenCtx streams by SM value");
}