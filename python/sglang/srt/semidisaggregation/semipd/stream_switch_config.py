"""
Configuration constants for dynamic stream switching in SemiPD mode.
"""

# Stream switching parameters
STREAM_SWITCH_IDLE_TIMEOUT = 10  # seconds: switch to decode-heavy when idle for this long
STREAM_SWITCH_CHECK_INTERVAL = 2  # seconds: how often to check for switching conditions
STREAM_SWITCH_MIN_INTERVAL = 30  # seconds: minimum time between switches
STREAM_SWITCH_IMMEDIATE_RESPONSE = True  # bool: whether to immediately switch back when new requests arrive

# Stream group indices
STREAM_GROUP_BALANCED = 1  # 80% prefill, 20% decode
STREAM_GROUP_DECODE_HEAVY = 2  # 10% prefill, 90% decode
STREAM_GROUP_PREFILL_HEAVY = 0  # 90% prefill, 10% decode

# Default stream group
DEFAULT_STREAM_GROUP = STREAM_GROUP_BALANCED
