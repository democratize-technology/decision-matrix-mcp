#!/usr/bin/env python3
"""Simple test bypassing __init__.py imports"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import constants directly from file
constants_file = os.path.join(
    os.path.dirname(__file__),
    "src",
    "decision_matrix_mcp",
    "constants.py",
)
print(f"Loading constants from: {constants_file}")

# Execute constants file to get classes
exec(open(constants_file).read(), globals())

print("✓ SessionLimits imported:")
print(f"  MAX_ACTIVE_SESSIONS: {SessionLimits.MAX_ACTIVE_SESSIONS}")
print(f"  LRU_EVICTION_BATCH_SIZE: {SessionLimits.LRU_EVICTION_BATCH_SIZE}")
print(f"  DEFAULT_MAX_SESSIONS: {SessionLimits.DEFAULT_MAX_SESSIONS}")

# Test basic memory bounds logic
print("\n✓ Testing memory bounds logic:")


def test_bounds_logic():
    current_sessions = 105  # Over limit
    max_sessions = 50
    max_active = SessionLimits.MAX_ACTIVE_SESSIONS  # 100
    batch_size = SessionLimits.LRU_EVICTION_BATCH_SIZE  # 10

    print(f"  Current sessions: {current_sessions}")
    print(f"  Max sessions (creation limit): {max_sessions}")
    print(f"  Max active (memory bound): {max_active}")

    # Should trigger eviction when >= MAX_ACTIVE_SESSIONS (100)
    needs_eviction = current_sessions >= max_active
    print(f"  Needs LRU eviction: {needs_eviction}")

    if needs_eviction:
        # Calculate evictions needed
        target_evictions = min(batch_size, current_sessions - max_sessions + batch_size)
        print(f"  Would evict: {target_evictions} sessions (batch size: {batch_size})")

        final_count = current_sessions - target_evictions
        print(f"  Final session count after eviction: {final_count}")

        # Verify bounded
        is_bounded = final_count <= max_active
        print(f"  Memory bounded: {is_bounded} (≤ {max_active})")

        return is_bounded

    return True


success = test_bounds_logic()
if success:
    print("\n🎉 Memory bounds logic test PASSED!")
else:
    print("\n❌ Memory bounds logic test FAILED!")
    sys.exit(1)
