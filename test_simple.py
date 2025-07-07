#!/usr/bin/env python3
"""
Simple Test Script for SVL Chatbot Core Components
Tests basic functionality without heavy AWS dependencies
"""

import sys
import asyncio
from datetime import datetime

# Add utils to path
sys.path.append('utils')

from conversation_manager import ConversationManager
from database_manager import DatabaseManager
from conversation_orchestrator import ConversationOrchestrator
from logger import get_logger

logger = get_logger("simple_test")

async def test_basic_components():
    """Test basic component initialization"""
    print("🧪 Testing Basic Component Initialization")
    
    try:
        # Test 1: Database Manager
        print("  ✓ Creating Database Manager...")
        db_manager = DatabaseManager()
        
        # Test 2: Conversation Manager  
        print("  ✓ Creating Conversation Manager...")
        conversation_manager = ConversationManager()
        
        # Test 3: Conversation Orchestrator
        print("  ✓ Creating Conversation Orchestrator...")
        orchestrator = ConversationOrchestrator(db_manager, conversation_manager)
        
        # Test 4: Initialize orchestrator
        print("  ✓ Initializing Orchestrator...")
        await orchestrator.initialize()
        
        # Test 5: Get system status
        print("  ✓ Getting System Status...")
        status = orchestrator.get_system_status()
        print(f"     System Health: {status['system_health']['status']}")
        print(f"     Active Conversations: {status['conversations']['active']}")
        
        # Test 6: Process a simple message (without Bedrock)
        print("  ✓ Testing Message Processing...")
        try:
            result = await orchestrator.process_user_message(
                user_id="test_user",
                conversation_id="test_conv_001", 
                message="Hello, I need help"
            )
            print(f"     Response: {result.get('response', 'No response')[:50]}...")
        except Exception as e:
            print(f"     ⚠️ Message processing failed (expected due to AWS): {str(e)[:50]}...")
        
        # Test 7: Cleanup
        print("  ✓ Shutting down gracefully...")
        await orchestrator.shutdown()
        
        print("✅ All basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_conversation_states():
    """Test conversation state management"""
    print("\n🧪 Testing Conversation State Management")
    
    try:
        # Import conversation engine components
        from conversation_engine import ConversationState, Intent
        from conversation_flow import ConversationContext
        
        # Test conversation context
        context = ConversationContext()
        print(f"  ✓ Initial state: {context.current_state}")
        print(f"  ✓ Session ID: {context.session_id}")
        
        # Test state transitions
        context.current_state = ConversationState.GREETING
        print(f"  ✓ State changed to: {context.current_state}")
        
        # Test data collection
        context.collected_data["test_field"] = "test_value"
        print(f"  ✓ Data collected: {len(context.collected_data)} fields")
        
        print("✅ Conversation state tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ State test failed: {e}")
        return False

async def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n🧪 Testing Performance Monitoring")
    
    try:
        from conversation_orchestrator import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test recording metrics
        monitor.record_conversation_start("test_conv_001")
        monitor.record_response_time(1.5)
        monitor.record_conversation_end("test_conv_001", True)
        
        # Test health status
        health = monitor.get_health_status()
        print(f"  ✓ System Status: {health['status']}")
        print(f"  ✓ Success Rate: {health['success_rate']}%")
        print(f"  ✓ Active Conversations: {health['active_conversations']}")
        
        print("✅ Performance monitoring tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

async def main():
    """Run all simple tests"""
    print("🚀 Starting Simple SVL Chatbot Tests")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # Run tests
    tests = [
        test_basic_components,
        test_conversation_states, 
        test_performance_monitoring
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    # Results
    execution_time = (datetime.now() - start_time).total_seconds()
    success_rate = (passed / total) * 100
    
    print("\n" + "=" * 50)
    print("📊 SIMPLE TEST RESULTS")
    print(f"Tests Run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Execution Time: {execution_time:.2f}s")
    
    if success_rate >= 80:
        print("🎉 Core system functionality is working correctly!")
    else:
        print("⚠️ Some issues detected - check logs for details")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}") 