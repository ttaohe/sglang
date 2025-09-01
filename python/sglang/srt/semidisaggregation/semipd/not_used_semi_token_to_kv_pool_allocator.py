"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Thread-safe memory pool allocator for Semi-PD mode.
This module provides thread-safe versions of TokenToKVPoolAllocator classes
to handle concurrent access from prefill and decode schedulers.
"""

import threading
from typing import TYPE_CHECKING
from functools import wraps
import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    AscendPagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)

def synchronized(debug_only=False):
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (not debug_only) or self.debug:
                with self.lock:
                    return func(self, *args, **kwargs)
            else:
                return True

        return wrapper

    return _decorator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class SemiTokenToKVPoolAllocator(TokenToKVPoolAllocator):
    """Thread-safe version of TokenToKVPoolAllocator for Semi-PD mode."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        self.lock = threading.RLock()
        super().__init__(size, dtype, device, kvcache, need_sort)
        

    @synchronized()
    def clear(self):
        return super().clear()

    @synchronized()
    def alloc(self, need_size: int):
        return super().alloc(need_size)

    @synchronized()
    def free(self, free_index: torch.Tensor):
        return super().free(free_index)

    @synchronized()
    def available_size(self):
        return super().available_size()

    @synchronized()
    def merge_and_sort_free(self):
        return super().merge_and_sort_free()

    @synchronized()
    def free_group_begin(self):
        return super().free_group_begin()

    @synchronized()
    def free_group_end(self):
        return super().free_group_end()

    @synchronized()
    def backup_state(self):
        return super().backup_state()

    @synchronized()
    def restore_state(self, state):
        return super().restore_state(state)


class SemiPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Thread-safe version of PagedTokenToKVPoolAllocator for Semi-PD mode."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        self.lock = threading.RLock()
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        

    @synchronized()
    def clear(self):
        return super().clear()

    @synchronized()
    def alloc(self, need_size: int):
        return super().alloc(need_size)

    @synchronized()
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        return super().alloc_extend(prefix_lens, seq_lens, last_loc, extend_num_tokens)

    @synchronized()
    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return super().alloc_decode(seq_lens, last_loc)

    @synchronized()
    def free(self, free_index: torch.Tensor):
        return super().free(free_index)

    @synchronized()
    def available_size(self):
        return super().available_size()

    @synchronized()
    def merge_and_sort_free(self):
        return super().merge_and_sort_free()

    @synchronized()
    def free_group_begin(self):
        return super().free_group_begin()

    @synchronized()
    def free_group_end(self):
        return super().free_group_end()

    @synchronized()
    def backup_state(self):
        return super().backup_state()

    @synchronized()
    def restore_state(self, state):
        return super().restore_state(state)


class SemiAscendPagedTokenToKVPoolAllocator(AscendPagedTokenToKVPoolAllocator):
    """Thread-safe version of AscendPagedTokenToKVPoolAllocator for Semi-PD mode."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.lock = threading.RLock()

    @synchronized()
    def clear(self):
        return super().clear()

    @synchronized()
    def alloc(self, need_size: int):
        return super().alloc(need_size)

    @synchronized()
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        return super().alloc_extend(prefix_lens, seq_lens, last_loc, extend_num_tokens)

    @synchronized()
    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        return super().alloc_decode(seq_lens, last_loc)

    @synchronized()
    def free(self, free_index: torch.Tensor):
        return super().free(free_index)

    @synchronized()
    def available_size(self):
        return super().available_size()

    @synchronized()
    def merge_and_sort_free(self):
        return super().merge_and_sort_free()

    @synchronized()
    def free_group_begin(self):
        return super().free_group_begin()

    @synchronized()
    def free_group_end(self):
        return super().free_group_end()

    @synchronized()
    def backup_state(self):
        return super().backup_state()

    @synchronized()
    def restore_state(self, state):
        return super().restore_state(state)


class SemiSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Thread-safe version of SWATokenToKVPoolAllocator for Semi-PD mode."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, size_swa, dtype, device, kvcache, need_sort)
        self.lock = threading.RLock()

    @synchronized()
    def clear(self):
        return super().clear()

    @synchronized()
    def alloc(self, need_size: int):
        return super().alloc(need_size)

    @synchronized()
    def free(self, free_index: torch.Tensor):
        return super().free(free_index)

    @synchronized()
    def full_available_size(self):
        return super().full_available_size()

    @synchronized()
    def swa_available_size(self):
        return super().swa_available_size()

    @synchronized()
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return super().translate_loc_from_full_to_swa(kv_indices)

    @synchronized()
    def free_group_begin(self):
        return super().free_group_begin()

    @synchronized()
    def free_group_end(self):
        return super().free_group_end()

    @synchronized()
    def backup_state(self):
        return super().backup_state()

    @synchronized()
    def restore_state(self, state):
        return super().restore_state(state)
