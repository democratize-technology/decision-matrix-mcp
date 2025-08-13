#!/usr/bin/env python3
"""Debug LRU eviction logic"""

import sys
sys.path.insert(0, 'src')

from decision_matrix_mcp.session_manager import SessionManager
from decision_matrix_mcp.constants import SessionLimits

def debug_lru_logic():
    print("=== Debugging LRU Logic ===")
    print(f"MAX_ACTIVE_SESSIONS: {SessionLimits.MAX_ACTIVE_SESSIONS}")
    print(f"LRU_EVICTION_BATCH_SIZE: {SessionLimits.LRU_EVICTION_BATCH_SIZE}")
    
    # Create manager with high max_sessions to avoid creation limit
    manager = SessionManager(max_sessions=200)
    print(f"Manager max_sessions: {manager.max_sessions}")
    
    print("\n1. Creating sessions...")
    # Create a smaller number first to debug
    sessions = []
    for i in range(10):
        session = manager.create_session(f"Topic {i}")
        sessions.append(session)
        print(f"   Created session {i}: {session.session_id[:8]}")
    
    print(f"   Session count: {len(manager.sessions)}")
    print(f"   Session order: {[sid[:8] for sid in manager.sessions.keys()]}")
    
    print("\n2. Accessing first 3 sessions to make them recently used...")
    for i in range(3):
        retrieved = manager.get_session(sessions[i].session_id)
        print(f"   Accessed session {i}: {sessions[i].session_id[:8]} -> {'Found' if retrieved else 'Not found'}")
    
    print(f"   Session order after access: {[sid[:8] for sid in manager.sessions.keys()]}")
    
    print("\n3. Manually triggering LRU eviction...")
    # Temporarily reduce MAX_ACTIVE_SESSIONS to test eviction
    original_max = SessionLimits.MAX_ACTIVE_SESSIONS
    SessionLimits.MAX_ACTIVE_SESSIONS = 8  # Force eviction
    
    try:
        manager._evict_lru_sessions()
        print(f"   Session count after eviction: {len(manager.sessions)}")
        print(f"   Session order after eviction: {[sid[:8] for sid in manager.sessions.keys()]}")
        
        # Check which sessions survived
        print("\n4. Checking surviving sessions...")
        for i, session in enumerate(sessions):
            exists = manager.get_session(session.session_id) is not None
            print(f"   Session {i} ({session.session_id[:8]}): {'Survived' if exists else 'Evicted'}")
            
    finally:
        # Restore original limit
        SessionLimits.MAX_ACTIVE_SESSIONS = original_max

if __name__ == "__main__":
    debug_lru_logic()