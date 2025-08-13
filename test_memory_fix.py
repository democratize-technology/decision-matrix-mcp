#!/usr/bin/env python3
"""Quick test to verify the memory leak fix works correctly"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decision_matrix_mcp.session_manager import SessionManager, SessionValidator
from decision_matrix_mcp.constants import SessionLimits
from datetime import datetime, timedelta, timezone

def test_basic_functionality():
    """Test that basic session operations still work"""
    print("Testing basic functionality...")
    
    manager = SessionManager(max_sessions=5)
    
    # Create a session
    session = manager.create_session("Test Topic")
    print(f"✓ Created session: {session.session_id[:8]}")
    
    # Retrieve session
    retrieved = manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    print("✓ Session retrieval works")
    
    # Test LRU tracking (accessing should move to end)
    assert session.session_id in manager.sessions
    print("✓ Session exists in OrderedDict")
    
    print("Basic functionality test PASSED!\n")

def test_lru_eviction():
    """Test that LRU eviction works when memory limits are reached"""
    print("Testing LRU eviction...")
    
    # Set limits to trigger eviction
    manager = SessionManager(max_sessions=3)
    
    sessions = []
    
    # Create sessions up to limit
    for i in range(3):
        session = manager.create_session(f"Topic {i}")
        sessions.append(session)
        print(f"✓ Created session {i}: {session.session_id[:8]}")
    
    # Access first session to make it recently used
    manager.get_session(sessions[0].session_id)
    print(f"✓ Accessed session 0 (should be most recently used)")
    
    # Create one more session - should trigger eviction
    session4 = manager.create_session("Topic 3")
    print(f"✓ Created session 3: {session4.session_id[:8]} (should trigger eviction)")
    
    # Check that we still have 3 sessions
    assert len(manager.sessions) == 3
    print(f"✓ Session count maintained at limit: {len(manager.sessions)}")
    
    # Check that session 0 (recently accessed) is still there
    assert manager.get_session(sessions[0].session_id) is not None
    print("✓ Most recently used session survived eviction")
    
    # Check that new session is there
    assert manager.get_session(session4.session_id) is not None
    print("✓ New session exists")
    
    print("LRU eviction test PASSED!\n")

def test_memory_bounds():
    """Test that memory is bounded at MAX_ACTIVE_SESSIONS"""
    print("Testing memory bounds...")
    
    # Create manager with high max_sessions but test against MAX_ACTIVE_SESSIONS
    manager = SessionManager(max_sessions=1000) 
    
    # Create sessions up to MAX_ACTIVE_SESSIONS
    sessions_created = 0
    try:
        for i in range(SessionLimits.MAX_ACTIVE_SESSIONS + 20):  # Try to exceed limit
            session = manager.create_session(f"Topic {i}")
            sessions_created += 1
            if i % 20 == 0:
                print(f"  Created {i} sessions...")
    except Exception as e:
        print(f"  Stopped at {sessions_created} sessions due to: {e}")
    
    # Verify we never exceed MAX_ACTIVE_SESSIONS
    assert len(manager.sessions) <= SessionLimits.MAX_ACTIVE_SESSIONS
    print(f"✓ Session count bounded at {len(manager.sessions)} <= {SessionLimits.MAX_ACTIVE_SESSIONS}")
    
    print("Memory bounds test PASSED!\n")

def test_stats_tracking():
    """Test that statistics tracking still works correctly"""
    print("Testing stats tracking...")
    
    manager = SessionManager(max_sessions=5)
    
    # Initial stats
    stats = manager.get_stats()
    assert stats["sessions_created"] == 0
    assert stats["active_sessions"] == 0
    print("✓ Initial stats correct")
    
    # Create sessions
    session1 = manager.create_session("Topic 1")
    session2 = manager.create_session("Topic 2")
    
    stats = manager.get_stats()
    assert stats["sessions_created"] == 2
    assert stats["active_sessions"] == 2
    print("✓ Stats after creation correct")
    
    # Remove session
    manager.remove_session(session1.session_id)
    
    stats = manager.get_stats()
    assert stats["sessions_cleaned"] == 1
    assert stats["active_sessions"] == 1
    print("✓ Stats after removal correct")
    
    print("Stats tracking test PASSED!\n")

if __name__ == "__main__":
    print("=== Testing Memory Leak Fix ===\n")
    
    try:
        test_basic_functionality()
        test_lru_eviction()
        test_memory_bounds() 
        test_stats_tracking()
        
        print("🎉 ALL TESTS PASSED! Memory leak fix is working correctly.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)