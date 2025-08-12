from math import log
import threading
import typing as _t
import logging

logger = logging.getLogger(__name__)

_DECODE_RUNNER_REF = None
_REF_LOCK = threading.Lock()
_REF_READY = threading.Event()


def register_decode_runner(runner: object) -> None:
    global _DECODE_RUNNER_REF
    with _REF_LOCK:
        _DECODE_RUNNER_REF = runner
        _REF_READY.set()
    logging.info("register_decode_runner success")


def wait_and_get_decode_runner(timeout: _t.Optional[float] = None) -> object:
    if not _REF_READY.wait(timeout=timeout):
        raise TimeoutError("decode runner not ready")
    return _DECODE_RUNNER_REF


def adopt_shared_runner(source_runner: object, target_runner: object) -> None:
    """
    Adopt decode runner's model and buffers into prefill runner and align
    derived fields to avoid re-initialization divergence.

    This is intentionally light-weight and side-effect free on "source_runner".
    """
    # Share model
    setattr(target_runner, "model", getattr(source_runner, "model"))
    logging.info("adopt_shared_runner share model success")

    # Share memory pools by reference to ensure full consistency
    if hasattr(source_runner, "req_to_token_pool"):
        try:
            target_runner.req_to_token_pool = source_runner.req_to_token_pool
            logging.info("adopt_shared_runner share req_to_token_pool ref success")
        except Exception:
            logging.info("adopt_shared_runner share req_to_token_pool ref failed")
    if hasattr(source_runner, "token_to_kv_pool_allocator"):
        try:
            target_runner.token_to_kv_pool_allocator = source_runner.token_to_kv_pool_allocator
            logging.info("adopt_shared_runner share token_to_kv_pool_allocator ref success")
        except Exception:
            logging.info("adopt_shared_runner share token_to_kv_pool_allocator ref failed")

    # Share token_to_kv_pool
    if hasattr(source_runner, "token_to_kv_pool"):
        try:
            target_runner.token_to_kv_pool = source_runner.token_to_kv_pool
            logging.info("adopt_shared_runner share token_to_kv_pool ref success")
        except Exception:
            logging.info("adopt_shared_runner share token_to_kv_pool ref failed")

    # Share kv_cache_dtype
    if hasattr(source_runner, "kv_cache_dtype"):
        try:
            target_runner.kv_cache_dtype = source_runner.kv_cache_dtype
            logging.info("adopt_shared_runner share kv_cache_dtype success")
        except Exception:
            pass
    elif hasattr(source_runner, "server_args") and hasattr(source_runner.server_args, "kv_cache_dtype"):
        try:
            target_runner.kv_cache_dtype = source_runner.server_args.kv_cache_dtype
            logging.info("adopt_shared_runner derive kv_cache_dtype from server_args success")
        except Exception:
            pass

    # Share req_to_token mapping if present
    if hasattr(source_runner, "req_to_token_pool") and hasattr(source_runner.req_to_token_pool, "req_to_token") and hasattr(target_runner, "req_to_token_pool"):
        try:
            target_runner.req_to_token_pool.req_to_token = source_runner.req_to_token_pool.req_to_token
            logging.info("adopt_shared_runner share req_to_token mapping success")
        except Exception:
            logging.info("adopt_shared_runner share req_to_token mapping failed")

    # Share max_total_num_tokens
    if hasattr(source_runner, "max_total_num_tokens"):
        try:
            target_runner.max_total_num_tokens = source_runner.max_total_num_tokens
            logging.info("adopt_shared_runner share max_total_num_tokens success")
        except Exception:
            logging.info("adopt_shared_runner share max_total_num_tokens failed")

    # Let ModelRunner finalize other dependent fields if method exists
    if hasattr(target_runner, "sync_after_adopt_shared_model"):
        try:
            target_runner.sync_after_adopt_shared_model()  # type: ignore[attr-defined]
        except Exception:
            pass


def adopt_from_registered_decode_if_needed(target_runner: object, timeout: _t.Optional[float] = None) -> None:
    try:
        source = wait_and_get_decode_runner(timeout=timeout)
        logging.info(f"source.__dir__(): {source.__dir__()}")
        adopt_shared_runner(source, target_runner)
        logging.info("adopt_from_registered_decode_if_needed success")
    except Exception:
        pass


