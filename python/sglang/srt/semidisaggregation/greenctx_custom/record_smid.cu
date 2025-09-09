// record_smid.cu
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


namespace py = pybind11;

/* ---------------- Kernel ---------------- */
__global__ void record_smid_kernel(int *hit, int sm_total)
{
    // 获取本 warp 所在 SM 的编号
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    // 只让 warp 里的第一个线程去做一次 atomic，避免多余冲突
    if (threadIdx.x == 0)
    {
        if (smid < (unsigned)sm_total)
            atomicAdd(&hit[smid], 1);
    }
}

/* ---------------- C++ 包装函数 ---------------- */
py::array_t<int> record_smid(int blocks_per_sm = 4,
                             int threads_per_block = 128,
                             int device_id = 0)
{
    // 1. 取得 GPU 上的 SM 总数
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    int sm_total = prop.multiProcessorCount;

    // 2. 在 device 端申请 hit[] 并清零
    int *d_hit;
    size_t bytes = sm_total * sizeof(int);
    cudaMalloc(&d_hit, bytes);
    cudaMemset(d_hit, 0, bytes);

    // 3. 设置 grid / block 规模
    int grid_blocks = sm_total * blocks_per_sm;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    record_smid_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(d_hit, sm_total);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(launch_err));

    cudaDeviceSynchronize();

    // 4. 拷贝回主机
    std::vector<int> h_hit(sm_total, 0);
    cudaMemcpy(h_hit.data(), d_hit, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_hit);

    return py::array_t<int>(h_hit.size(), h_hit.data());  // 不传 base，对数据做一次 copy

}

/* ---------------- pybind11 导出 ---------------- */
PYBIND11_MODULE(smid_recorder, m)
{
    m.doc() = "CUDA SM-usage recorder (GreenCtx / MPS 验证工具)";
    m.def("record",
          &record_smid,
          py::arg("blocks_per_sm") = 4,
          py::arg("threads_per_block") = 128,
          py::arg("device_id") = 0,
          R"pbdoc(
          统计当前 CUDA Context 在一次 kernel 启动里实际用到哪些 SM。

          返回值:  长度 = 设备 SM 总数的一维 numpy.int32 数组。
                    hit[i] > 0 表示第 i 号 SM 至少驻留过一个 block。
          )pbdoc");
}
