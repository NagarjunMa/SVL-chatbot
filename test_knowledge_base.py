#!/usr/bin/env python3
"""
Knowledge Base Testing Script
Standalone script to test the SVL knowledge base system
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add utils to path
sys.path.append('.')

def main():
    parser = argparse.ArgumentParser(description="Test SVL Knowledge Base System")
    parser.add_argument("--init", action="store_true", help="Initialize knowledge base")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--test", action="store_true", help="Run functionality tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--health", action="store_true", help="Check system health")
    parser.add_argument("--query", type=str, help="Test a specific query")
    parser.add_argument("--query-type", type=str, default="general", 
                       choices=["general", "sop", "faq", "process", "contact"], 
                       help="Query type for test query")
    parser.add_argument("--report", type=str, help="Generate report file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Import after configuring logging
        from utils.knowledge_base_initializer import (
            KnowledgeBaseInitializer, 
            quick_initialize, 
            full_test_suite, 
            health_check
        )
        from utils.knowledge_base import KnowledgeBase
        
        print("🚗 SVL Knowledge Base Testing System")
        print("=" * 50)
        
        # Health check
        if args.health:
            print("\n📊 Running health check...")
            health_result = health_check()
            print(f"Status: {health_result['status']}")
            print(f"Ready: {health_result['ready']}")
            if 'stats' in health_result:
                stats = health_result['stats']
                search_stats = stats.get('search_engine', {})
                print(f"Index size: {search_stats.get('embeddings', {}).get('index_size', 0)} embeddings")
            print()
        
        # Initialize knowledge base
        if args.init:
            print("\n🔧 Initializing knowledge base...")
            if args.performance:
                print("Running full test suite with performance testing...")
                results = full_test_suite(force_rebuild=args.force_rebuild)
            else:
                print("Running quick initialization...")
                results = quick_initialize(force_rebuild=args.force_rebuild)
            
            print(f"Status: {results.get('overall_status', 'unknown')}")
            
            if args.report:
                initializer = KnowledgeBaseInitializer()
                report = initializer.generate_report(results, args.report)
                print(f"Report saved to: {args.report}")
            
            # Print summary
            init_result = results.get('initialization', {})
            if init_result:
                print(f"Initialization time: {init_result.get('initialization_time', 0):.2f}s")
                build_details = init_result.get('build_details', {})
                if build_details:
                    print(f"Documents processed: {build_details.get('document_count', 0)}")
                    print(f"Chunks created: {build_details.get('chunk_count', 0)}")
                    print(f"Embeddings generated: {build_details.get('embedding_count', 0)}")
            
            validation = results.get('validation', {})
            if validation:
                print(f"Test success rate: {validation.get('success_rate', 0):.1%}")
            
            print()
        
        # Test specific query
        if args.query:
            print(f"\n🔍 Testing query: '{args.query}'")
            print(f"Query type: {args.query_type}")
            
            try:
                kb = KnowledgeBase()
                if not kb.is_ready():
                    print("Knowledge base not ready. Please run --init first.")
                    return
                
                result = kb.query_knowledge_base(
                    user_query=args.query,
                    query_type=args.query_type,
                    include_context=True
                )
                
                print(f"Status: {result['status']}")
                print(f"Results found: {result.get('total_results', 0)}")
                print(f"Search time: {result.get('search_time', 0):.3f}s")
                
                if result.get('total_results', 0) > 0:
                    print("\nTop results:")
                    for i, res in enumerate(result['results'][:3], 1):
                        print(f"  {i}. {res['document']['filename']} (Score: {res['relevance_score']:.3f})")
                        print(f"     {res['text'][:100]}...")
                        print()
                
                if result.get('context'):
                    print("Context for AI:")
                    print(result['context'][:300] + "..." if len(result['context']) > 300 else result['context'])
                
            except Exception as e:
                print(f"Query failed: {e}")
            
            print()
        
        # Run functional tests only
        if args.test and not args.init:
            print("\n🧪 Running functional tests...")
            try:
                kb = KnowledgeBase()
                if not kb.is_ready():
                    print("Knowledge base not ready. Please run --init first.")
                    return
                
                # Simple test queries
                test_queries = [
                    {"query": "How to report stolen vehicle", "type": "process"},
                    {"query": "Contact information", "type": "contact"},
                    {"query": "Pricing information", "type": "faq"},
                    {"query": "Emergency procedures", "type": "sop"}
                ]
                
                successful = 0
                for i, test in enumerate(test_queries, 1):
                    try:
                        result = kb.query_knowledge_base(
                            user_query=test["query"],
                            query_type=test["type"],
                            max_results=1
                        )
                        success = result["status"] == "success" and result.get("total_results", 0) > 0
                        if success:
                            successful += 1
                        
                        status_emoji = "✅" if success else "❌"
                        print(f"  {status_emoji} Test {i}: {test['query']} ({test['type']})")
                        
                    except Exception as e:
                        print(f"  ❌ Test {i}: {test['query']} - Error: {e}")
                
                print(f"\nTest Results: {successful}/{len(test_queries)} passed ({successful/len(test_queries):.1%})")
                
            except Exception as e:
                print(f"Testing failed: {e}")
            
            print()
        
        # Show available commands if no action specified
        if not any([args.init, args.test, args.health, args.query]):
            print("\n📋 Available commands:")
            print("  --health              Check system health")
            print("  --init                Initialize knowledge base")
            print("  --init --performance  Initialize with full performance testing")
            print("  --test                Run functional tests")
            print("  --query 'your query'  Test a specific query")
            print("  --force-rebuild       Force rebuild of index")
            print("  --report FILE         Generate report file")
            print("  --verbose             Enable verbose logging")
            print("\nExample usage:")
            print("  python test_knowledge_base.py --init --performance --report kb_report.txt")
            print("  python test_knowledge_base.py --query 'How to report stolen vehicle' --query-type process")
            print("  python test_knowledge_base.py --health")
            print()
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed all dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 