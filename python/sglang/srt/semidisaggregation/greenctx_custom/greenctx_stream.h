#include <vector>

// 结果结构体，类似于 Python 的 NamedTuple
struct SMAllocationResult {
    int64_t streamA_ptr;
    int64_t streamB_ptr;
    int64_t sm_a;
    int64_t sm_b;
    float actual_percent_a;
    float actual_percent_b;
};

SMAllocationResult create_greenctx_stream_by_percent(float smA, float smB, int device);
std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device);

