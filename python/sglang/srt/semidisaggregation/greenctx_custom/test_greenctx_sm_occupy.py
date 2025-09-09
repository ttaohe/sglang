import pytest
import torch
import torch.nn.functional as F
from greenctx import(
    create_greenctx_stream_by_value,
)
import numpy as np
import smid_recorder


def _check_active_sms(expect_cnt: int):
    """
    Helper: 在当前 stream 中 launch record_smid kernel，
    返回活跃 SM id 列表并断言数量是否符合预期。
    """
    hits = smid_recorder.record()          # kernel 在“当前 stream”执行
    active = np.nonzero(hits)[0]
    print(f"  active_sm = {active.tolist()}")
    assert len(active) == expect_cnt, (
        f"Expect {expect_cnt} SM, but got {len(active)}")


def test_green_ctx():
    property = torch.cuda.get_device_properties(0)
    stream_group = create_greenctx_stream_by_value(46, 46, 0)
    A = torch.randn(4096, 4096).cuda()
    B = torch.randn(4096, 4096).cuda()
    with torch.cuda.stream(torch.cuda.get_stream_from_external(stream_group[0])):
        for _ in range(1):
            hits = smid_recorder.record() 
            active = np.nonzero(hits)[0]
            print(f"sms: {len(active)}, active_sm = {active.tolist()}")
            assert len(active) == 46, (
                f"Expect {46} SM, but got {len(active)}")
        # torch.matmul(A, B)


if __name__ == "__main__":
    # pytest.main([__file__])
    test_green_ctx()
