"""
Stream synchronization manager for SemiPD mode.
Handles cross-thread communication for stream switching.
"""

import threading
import time
import logging
from typing import Optional

from .pdmux_context import (
    get_current_stream_idx,
    get_prefill_stream,
    get_decode_stream,
    get_stream_group_name,
    set_current_stream_idx,
)
from .stream_switch_config import (
    STREAM_GROUP_BALANCED,
    STREAM_GROUP_DECODE_HEAVY,
    STREAM_SWITCH_IDLE_TIMEOUT,
    STREAM_SWITCH_CHECK_INTERVAL,
    STREAM_SWITCH_MIN_INTERVAL,
)

logger = logging.getLogger(__name__)


class StreamSyncManager:
    """Manages synchronized stream switching between prefill and decode schedulers."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._last_switch_time = 0
        self._current_stream_idx = STREAM_GROUP_BALANCED
        self._prefill_scheduler = None
        self._decode_scheduler = None
        self._monitoring = False
        self._monitor_thread = None
        
    def register_schedulers(self, prefill_scheduler=None, decode_scheduler=None):
        """Register prefill and decode schedulers for synchronization."""
        with self._lock:
            if prefill_scheduler:
                self._prefill_scheduler = prefill_scheduler
            if decode_scheduler:
                self._decode_scheduler = decode_scheduler
                
    def can_switch(self) -> bool:
        """Check if enough time has passed since last switch."""
        current_time = time.time()
        return (current_time - self._last_switch_time) >= STREAM_SWITCH_MIN_INTERVAL
        
    def switch_to_decode_heavy(self) -> bool:
        """Switch to decode-heavy configuration."""
        with self._lock:
            if not self.can_switch():
                return False
                
            if self._current_stream_idx == STREAM_GROUP_DECODE_HEAVY:
                return False
                
            self._current_stream_idx = STREAM_GROUP_DECODE_HEAVY
            self._last_switch_time = time.time()
            
            # Update both schedulers
            self._update_both_schedulers()
            
            logger.info(f"System switched to decode-heavy configuration")
            return True
            
    def switch_to_balanced(self) -> bool:
        """Switch to balanced configuration."""
        with self._lock:
            if not self.can_switch():
                return False
                
            if self._current_stream_idx == STREAM_GROUP_BALANCED:
                return False
                
            self._current_stream_idx = STREAM_GROUP_BALANCED
            self._last_switch_time = time.time()
            
            # Update both schedulers
            self._update_both_schedulers()
            
            logger.info(f"System switched to balanced configuration")
            return True
            
    def _update_both_schedulers(self):
        """Update both prefill and decode schedulers with new stream configuration."""
        # Update global stream index
        set_current_stream_idx(self._current_stream_idx)
        
        # Update prefill scheduler
        if self._prefill_scheduler:
            try:
                prefill_stream = get_prefill_stream()
                self._prefill_scheduler.set_forward_stream(prefill_stream)
                logger.info(f"Prefill scheduler updated to use {get_stream_group_name(self._current_stream_idx)} prefill stream")
            except Exception as e:
                logger.error(f"Error updating prefill scheduler: {e}")
                
        # Update decode scheduler
        if self._decode_scheduler:
            try:
                decode_stream = get_decode_stream()
                self._decode_scheduler.set_forward_stream(decode_stream)
                logger.info(f"Decode scheduler updated to use {get_stream_group_name(self._current_stream_idx)} decode stream")
            except Exception as e:
                logger.error(f"Error updating decode scheduler: {e}")
                
    def start_monitoring(self, decode_scheduler):
        """Start monitoring for dynamic switching."""
        if self._monitoring:
            return
            
        self._decode_scheduler = decode_scheduler
        self._monitoring = True
        
        def _monitor():
            last_check_time = time.time()
            last_request_time = time.time()
            
            while self._monitoring:
                try:
                    time.sleep(STREAM_SWITCH_CHECK_INTERVAL)
                    
                    current_time = time.time()
                    
                    # Check for new activity
                    has_waiting_requests = len(decode_scheduler.waiting_queue) > 0
                    has_scheduled_prefill = len(decode_scheduler.scheduled_prefill_batches) > 0
                    has_active_decode = not decode_scheduler.running_batch.is_empty()
                    
                    if has_waiting_requests or has_scheduled_prefill:
                        last_request_time = current_time
                    
                    # Check if we should switch to decode-heavy
                    # Only switch when no active decode requests and no pending work
                    if (not has_waiting_requests and 
                        not has_scheduled_prefill and
                        has_active_decode and
                        current_time - last_request_time >= STREAM_SWITCH_IDLE_TIMEOUT and
                        self._current_stream_idx != STREAM_GROUP_DECODE_HEAVY):
                        
                        self.switch_to_decode_heavy()
                    
                    # Check if we should switch back to balanced
                    elif (not has_active_decode or 
                          (has_waiting_requests and self._current_stream_idx != STREAM_GROUP_BALANCED)
                        ):
                        
                        self.switch_to_balanced()
                        
                except Exception as e:
                    logger.error(f"Error in stream monitoring: {e}")
                    
        self._monitor_thread = threading.Thread(
            target=_monitor,
            name="stream_sync_monitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Stream synchronization monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def get_current_config(self) -> str:
        """Get current configuration name."""
        return get_stream_group_name(self._current_stream_idx)


# Global instance
stream_sync_manager = StreamSyncManager()
