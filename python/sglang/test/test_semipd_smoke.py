import os
import sys
import time
import pytest
from sglang.srt.server_args import ServerArgs

# --------- Simple local variables (edit as needed) ---------
MODEL_PATH = "/mnt/DeepSeek-R1-Distill-Qwen-32B"  # 修改为本地已下载模型的目录
TP_SIZE = 4
PORT = 33000
TIMEOUT_S = 180
# -----------------------------------------------------------


def _skip_with_message(msg: str):
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(msg)
    else:
        print(f"[SKIP] {msg}")
        sys.exit(0)


def _require_cuda_and_devices(tp_size: int):
    try:
        import torch
    except Exception:
        _skip_with_message("torch not available")

    if not torch.cuda.is_available():
        _skip_with_message("CUDA not available")
    if torch.cuda.device_count() < tp_size:
        _skip_with_message(
            f"Not enough GPUs: need {tp_size}, have {torch.cuda.device_count()}"
        )


def _run_smoke(model_path: str, tp_size: int, port: int):
    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        server_args=ServerArgs(
            model_path=model_path,
            engine_mode="semipd",
            device="cuda",
            dp_size=1,
            tp_size=tp_size,
            pp_size=1,
            load_format="dummy",
            skip_tokenizer_init=True,
            disable_cuda_graph=True,
            allow_auto_truncate=True,
            host="127.0.0.1",
            port=port,
            enable_metrics=False,
            log_level="error",
        )
    )

    try:
        t0 = time.time()
        server_info = engine.get_server_info()
        assert "version" in server_info
        assert (
            "max_total_num_tokens" in server_info
            or "max_req_input_len" in server_info
        )
        print(
            f"[OK] semipd smoke passed in {time.time() - t0:.2f}s, tp_size={tp_size}"
        )
    finally:
        engine.shutdown()


@pytest.mark.timeout(TIMEOUT_S)
def test_semipd_engine_smoke_dp1_tp4():
    if MODEL_PATH == "/path/to/your/local/model" or not os.path.exists(MODEL_PATH):
        _skip_with_message("Please set MODEL_PATH in this file to a valid local model path")
    _require_cuda_and_devices(TP_SIZE)
    _run_smoke(MODEL_PATH, TP_SIZE, PORT)


if __name__ == "__main__":
    if MODEL_PATH == "/path/to/your/local/model" or not os.path.exists(MODEL_PATH):
        _skip_with_message("Please set MODEL_PATH in this file to a valid local model path")
    _require_cuda_and_devices(TP_SIZE)
    _run_smoke(MODEL_PATH, TP_SIZE, PORT)


