"""
Knowledge Base Initializer and Performance Testing
Comprehensive setup, testing, and optimization utilities
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio

from utils.knowledge_base import KnowledgeBase
from utils.logger import get_logger

logger = get_logger("knowledge_base_initializer")

class KnowledgeBaseInitializer:
    """
    Comprehensive knowledge base initialization and testing utility
    Handles setup, performance optimization, and validation
    """
    
    def __init__(self, 
                 s3_bucket_name: str = None,
                 index_storage_path: str = "./knowledge_base_index",
                 aws_region: str = "us-east-1"):
        """
        Initialize the knowledge base initializer
        
        Args:
            s3_bucket_name: S3 bucket for documents
            index_storage_path: Local index storage path
            aws_region: AWS region
        """
        self.s3_bucket_name = s3_bucket_name or os.getenv(
            'KNOWLEDGE_BASE_S3_BUCKET', 
            'svl-knowledge-base'
        )
        self.index_storage_path = index_storage_path
        self.aws_region = aws_region
        
        self.knowledge_base = None
        
        # Test queries for validation
        self.test_queries = [
            {"query": "How do I report a stolen vehicle?", "type": "process", "expected_category": "sop"},
            {"query": "What is the recovery process?", "type": "process", "expected_category": "process_docs"},
            {"query": "How much does the service cost?", "type": "faq", "expected_category": "faq"},
            {"query": "Who can I contact for help?", "type": "contact", "expected_category": "contact"},
            {"query": "What documents do I need?", "type": "general", "expected_category": "sop"},
            {"query": "Emergency procedures for stolen vehicle", "type": "sop", "expected_category": "sop"},
            {"query": "Mobile app instructions", "type": "process", "expected_category": "process_docs"},
            {"query": "Customer frequently asked questions", "type": "faq", "expected_category": "faq"},
            {"query": "Pricing information details", "type": "faq", "expected_category": "faq"},
            {"query": "Contact information emergency", "type": "contact", "expected_category": "contact"}
        ]
        
        # Performance metrics
        self.performance_metrics = {
            "initialization_time": 0,
            "index_build_time": 0,
            "average_query_time": 0,
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "cache_hit_rate": 0,
            "search_accuracy": 0
        }
    
    def initialize_and_test(self, force_rebuild: bool = False, run_performance_tests: bool = True) -> Dict[str, Any]:
        """
        Complete initialization and testing pipeline
        
        Args:
            force_rebuild: Whether to force rebuild the index
            run_performance_tests: Whether to run performance tests
            
        Returns:
            Comprehensive results
        """
        logger.info("Starting comprehensive knowledge base initialization and testing...")
        start_time = time.time()
        
        results = {
            "initialization": {},
            "validation": {},
            "performance": {},
            "recommendations": [],
            "overall_status": "unknown"
        }
        
        try:
            # Step 1: Initialize knowledge base
            logger.info("Step 1: Initializing knowledge base...")
            init_result = self._initialize_knowledge_base(force_rebuild)
            results["initialization"] = init_result
            
            if init_result["status"] != "success":
                results["overall_status"] = "failed"
                results["recommendations"].append("Knowledge base initialization failed - check AWS credentials and S3 bucket")
                return results
            
            # Step 2: Validate functionality
            logger.info("Step 2: Validating functionality...")
            validation_result = self._validate_functionality()
            results["validation"] = validation_result
            
            # Step 3: Performance testing
            if run_performance_tests:
                logger.info("Step 3: Running performance tests...")
                perf_result = self._run_performance_tests()
                results["performance"] = perf_result
            
            # Step 4: Generate recommendations
            recommendations = self._generate_recommendations(results)
            results["recommendations"] = recommendations
            
            # Determine overall status
            if validation_result.get("success_rate", 0) > 0.8:
                results["overall_status"] = "success"
            elif validation_result.get("success_rate", 0) > 0.5:
                results["overall_status"] = "partial_success"
            else:
                results["overall_status"] = "needs_attention"
            
            total_time = time.time() - start_time
            results["total_time"] = total_time
            
            logger.info(f"Knowledge base initialization and testing completed in {total_time:.2f} seconds")
            logger.info(f"Overall status: {results['overall_status']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Initialization and testing failed: {e}")
            results["overall_status"] = "error"
            results["error"] = str(e)
            return results
    
    def _initialize_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the knowledge base system"""
        try:
            init_start = time.time()
            
            # Create knowledge base instance
            self.knowledge_base = KnowledgeBase(
                s3_bucket_name=self.s3_bucket_name,
                index_storage_path=self.index_storage_path,
                aws_region=self.aws_region
            )
            
            # Initialize
            result = self.knowledge_base.initialize_knowledge_base(force_rebuild=force_rebuild)
            
            init_time = time.time() - init_start
            self.performance_metrics["initialization_time"] = init_time
            
            # Get additional stats
            if result["status"] == "success":
                stats = self.knowledge_base.get_knowledge_base_stats()
                build_result = result.get("build_result", {})
                
                self.performance_metrics["total_documents_processed"] = build_result.get("document_count", 0)
                self.performance_metrics["total_chunks_created"] = build_result.get("chunk_count", 0)
                self.performance_metrics["total_embeddings_generated"] = build_result.get("embedding_count", 0)
                self.performance_metrics["index_build_time"] = build_result.get("build_time", 0)
            
            return {
                "status": result["status"],
                "message": result["message"],
                "initialization_time": init_time,
                "s3_bucket": self.s3_bucket_name,
                "index_path": self.index_storage_path,
                "build_details": result.get("build_result", {}),
                "ready": self.knowledge_base.is_ready() if self.knowledge_base else False
            }
            
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "initialization_time": 0,
                "ready": False
            }
    
    def _validate_functionality(self) -> Dict[str, Any]:
        """Validate knowledge base functionality with test queries"""
        if not self.knowledge_base or not self.knowledge_base.is_ready():
            return {
                "status": "not_ready",
                "success_rate": 0,
                "test_results": []
            }
        
        test_results = []
        successful_tests = 0
        
        for i, test_case in enumerate(self.test_queries):
            try:
                start_time = time.time()
                
                # Query knowledge base
                result = self.knowledge_base.query_knowledge_base(
                    user_query=test_case["query"],
                    query_type=test_case["type"],
                    include_context=True,
                    max_results=3
                )
                
                query_time = time.time() - start_time
                
                # Evaluate results
                success = result["status"] == "success" and result.get("total_results", 0) > 0
                
                # Check if results match expected category
                category_match = False
                if success and result.get("results"):
                    for search_result in result["results"]:
                        if search_result.get("document", {}).get("category") == test_case["expected_category"]:
                            category_match = True
                            break
                
                if success:
                    successful_tests += 1
                
                test_result = {
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "query_type": test_case["type"],
                    "expected_category": test_case["expected_category"],
                    "success": success,
                    "category_match": category_match,
                    "results_count": result.get("total_results", 0),
                    "confidence": result.get("confidence", 0),
                    "query_time": query_time,
                    "from_cache": result.get("from_cache", False)
                }
                
                test_results.append(test_result)
                
            except Exception as e:
                logger.error(f"Test query {i+1} failed: {e}")
                test_results.append({
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "success": False,
                    "error": str(e),
                    "query_time": 0
                })
        
        success_rate = successful_tests / len(self.test_queries) if self.test_queries else 0
        
        return {
            "status": "completed",
            "total_tests": len(self.test_queries),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "test_results": test_results,
            "average_query_time": sum(t.get("query_time", 0) for t in test_results) / len(test_results)
        }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests"""
        if not self.knowledge_base or not self.knowledge_base.is_ready():
            return {"status": "not_ready", "message": "Knowledge base not ready for performance testing"}
        
        perf_results = {
            "query_performance": {},
            "cache_performance": {},
            "concurrent_performance": {},
            "memory_usage": {}
        }
        
        try:
            # Test 1: Query performance with different query types
            logger.info("Testing query performance...")
            query_perf = self._test_query_performance()
            perf_results["query_performance"] = query_perf
            
            # Test 2: Cache performance
            logger.info("Testing cache performance...")
            cache_perf = self._test_cache_performance()
            perf_results["cache_performance"] = cache_perf
            
            # Test 3: Concurrent queries (simplified)
            logger.info("Testing concurrent query handling...")
            concurrent_perf = self._test_concurrent_performance()
            perf_results["concurrent_performance"] = concurrent_perf
            
            # Test 4: Memory usage
            logger.info("Checking memory usage...")
            memory_usage = self._check_memory_usage()
            perf_results["memory_usage"] = memory_usage
            
            return {
                "status": "completed",
                "results": perf_results,
                "overall_performance": self._evaluate_performance(perf_results)
            }
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _test_query_performance(self) -> Dict[str, Any]:
        """Test query performance across different query types"""
        query_types = ["general", "sop", "faq", "process", "contact"]
        results = {}
        
        for query_type in query_types:
            times = []
            test_query = f"What information is available about {query_type}?"
            
            # Run multiple queries for average
            for _ in range(3):
                start_time = time.time()
                result = self.knowledge_base.query_knowledge_base(
                    user_query=test_query,
                    query_type=query_type,
                    max_results=5
                )
                query_time = time.time() - start_time
                times.append(query_time)
            
            results[query_type] = {
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "queries_tested": len(times)
            }
        
        return results
    
    def _test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance with repeated queries"""
        test_query = "How do I report a stolen vehicle?"
        
        # First query (cache miss)
        start_time = time.time()
        result1 = self.knowledge_base.query_knowledge_base(
            user_query=test_query,
            query_type="process"
        )
        first_query_time = time.time() - start_time
        
        # Second query (should be cache hit)
        start_time = time.time()
        result2 = self.knowledge_base.query_knowledge_base(
            user_query=test_query,
            query_type="process"
        )
        second_query_time = time.time() - start_time
        
        # Calculate cache efficiency
        cache_efficiency = (first_query_time - second_query_time) / first_query_time if first_query_time > 0 else 0
        
        return {
            "first_query_time": first_query_time,
            "second_query_time": second_query_time,
            "cache_efficiency": cache_efficiency,
            "cache_hit_detected": result2.get("from_cache", False),
            "speed_improvement": f"{cache_efficiency * 100:.1f}%" if cache_efficiency > 0 else "No improvement"
        }
    
    def _test_concurrent_performance(self) -> Dict[str, Any]:
        """Test handling of concurrent queries (simplified simulation)"""
        queries = [
            "How to report stolen vehicle?",
            "What is the recovery process?",
            "Contact information please",
            "Emergency procedures",
            "Pricing information"
        ]
        
        start_time = time.time()
        results = []
        
        # Simulate concurrent queries (sequential for simplicity)
        for query in queries:
            query_start = time.time()
            result = self.knowledge_base.query_knowledge_base(
                user_query=query,
                query_type="general"
            )
            query_time = time.time() - query_start
            results.append({
                "query": query,
                "time": query_time,
                "success": result["status"] == "success"
            })
        
        total_time = time.time() - start_time
        
        return {
            "total_queries": len(queries),
            "total_time": total_time,
            "average_time_per_query": total_time / len(queries),
            "successful_queries": sum(1 for r in results if r["success"]),
            "query_details": results
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage of knowledge base components"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get knowledge base stats
            kb_stats = self.knowledge_base.get_knowledge_base_stats()
            
            return {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "virtual_memory_mb": memory_info.vms / 1024 / 1024,
                "index_size": kb_stats.get("search_engine", {}).get("embeddings", {}).get("index_size", 0),
                "cache_size": kb_stats.get("search_engine", {}).get("embeddings", {}).get("cache_size", 0),
                "status": "measured"
            }
        except ImportError:
            return {
                "status": "psutil_not_available",
                "message": "Install psutil for detailed memory metrics"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _evaluate_performance(self, perf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall performance and provide ratings"""
        evaluation = {
            "query_speed": "unknown",
            "cache_efficiency": "unknown",
            "concurrent_handling": "unknown",
            "overall_rating": "unknown"
        }
        
        try:
            # Evaluate query speed
            query_perf = perf_results.get("query_performance", {})
            if query_perf:
                avg_times = [q.get("average_time", 999) for q in query_perf.values()]
                overall_avg = sum(avg_times) / len(avg_times) if avg_times else 999
                
                if overall_avg < 1.0:
                    evaluation["query_speed"] = "excellent"
                elif overall_avg < 2.0:
                    evaluation["query_speed"] = "good"
                elif overall_avg < 5.0:
                    evaluation["query_speed"] = "acceptable"
                else:
                    evaluation["query_speed"] = "needs_improvement"
            
            # Evaluate cache efficiency
            cache_perf = perf_results.get("cache_performance", {})
            cache_efficiency = cache_perf.get("cache_efficiency", 0)
            
            if cache_efficiency > 0.5:
                evaluation["cache_efficiency"] = "excellent"
            elif cache_efficiency > 0.3:
                evaluation["cache_efficiency"] = "good"
            elif cache_efficiency > 0.1:
                evaluation["cache_efficiency"] = "acceptable"
            else:
                evaluation["cache_efficiency"] = "needs_improvement"
            
            # Evaluate concurrent handling
            concurrent_perf = perf_results.get("concurrent_performance", {})
            success_rate = concurrent_perf.get("successful_queries", 0) / concurrent_perf.get("total_queries", 1)
            
            if success_rate >= 1.0:
                evaluation["concurrent_handling"] = "excellent"
            elif success_rate >= 0.8:
                evaluation["concurrent_handling"] = "good"
            elif success_rate >= 0.6:
                evaluation["concurrent_handling"] = "acceptable"
            else:
                evaluation["concurrent_handling"] = "needs_improvement"
            
            # Overall rating
            ratings = ["excellent", "good", "acceptable", "needs_improvement"]
            rating_scores = {rating: i for i, rating in enumerate(ratings)}
            
            scores = [
                rating_scores.get(evaluation["query_speed"], 3),
                rating_scores.get(evaluation["cache_efficiency"], 3),
                rating_scores.get(evaluation["concurrent_handling"], 3)
            ]
            
            avg_score = sum(scores) / len(scores)
            evaluation["overall_rating"] = ratings[int(avg_score)]
            
        except Exception as e:
            logger.warning(f"Performance evaluation failed: {e}")
        
        return evaluation
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        try:
            # Initialization recommendations
            init_result = results.get("initialization", {})
            if init_result.get("status") != "success":
                recommendations.append("Fix knowledge base initialization issues before proceeding")
                return recommendations
            
            # Validation recommendations
            validation = results.get("validation", {})
            success_rate = validation.get("success_rate", 0)
            
            if success_rate < 0.5:
                recommendations.append("Low query success rate - review document processing and embedding quality")
            elif success_rate < 0.8:
                recommendations.append("Moderate query success rate - consider optimizing similarity thresholds")
            
            # Performance recommendations
            performance = results.get("performance", {})
            if performance.get("status") == "completed":
                perf_eval = performance.get("overall_performance", {})
                
                if perf_eval.get("query_speed") == "needs_improvement":
                    recommendations.append("Query performance is slow - consider optimizing index or reducing chunk sizes")
                
                if perf_eval.get("cache_efficiency") == "needs_improvement":
                    recommendations.append("Cache efficiency is low - increase cache size or improve cache management")
                
                if perf_eval.get("concurrent_handling") == "needs_improvement":
                    recommendations.append("Concurrent query handling needs improvement - consider async processing")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Knowledge base is performing well - consider monitoring and periodic reindexing")
            
            # Add performance optimization tips
            recommendations.extend([
                "Consider using Lambda layers for faster cold starts",
                "Monitor S3 costs and implement lifecycle policies for old embeddings",
                "Set up CloudWatch monitoring for query performance",
                "Consider using Bedrock Agents for production deployment"
            ])
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Generate a comprehensive test report"""
        report_lines = [
            "=" * 80,
            "SVL KNOWLEDGE BASE INITIALIZATION AND TESTING REPORT",
            "=" * 80,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"S3 Bucket: {self.s3_bucket_name}",
            f"Index Path: {self.index_storage_path}",
            "",
            "OVERALL STATUS: " + results.get("overall_status", "unknown").upper(),
            ""
        ]
        
        # Initialization results
        init_result = results.get("initialization", {})
        report_lines.extend([
            "INITIALIZATION RESULTS:",
            "-" * 40,
            f"Status: {init_result.get('status', 'unknown')}",
            f"Initialization Time: {init_result.get('initialization_time', 0):.2f} seconds",
            f"Documents Processed: {self.performance_metrics.get('total_documents_processed', 0)}",
            f"Chunks Created: {self.performance_metrics.get('total_chunks_created', 0)}",
            f"Embeddings Generated: {self.performance_metrics.get('total_embeddings_generated', 0)}",
            ""
        ])
        
        # Validation results
        validation = results.get("validation", {})
        if validation:
            report_lines.extend([
                "VALIDATION RESULTS:",
                "-" * 40,
                f"Total Tests: {validation.get('total_tests', 0)}",
                f"Successful Tests: {validation.get('successful_tests', 0)}",
                f"Success Rate: {validation.get('success_rate', 0):.1%}",
                f"Average Query Time: {validation.get('average_query_time', 0):.3f} seconds",
                ""
            ])
        
        # Performance results
        performance = results.get("performance", {})
        if performance.get("status") == "completed":
            perf_eval = performance.get("overall_performance", {})
            report_lines.extend([
                "PERFORMANCE EVALUATION:",
                "-" * 40,
                f"Query Speed: {perf_eval.get('query_speed', 'unknown')}",
                f"Cache Efficiency: {perf_eval.get('cache_efficiency', 'unknown')}",
                f"Concurrent Handling: {perf_eval.get('concurrent_handling', 'unknown')}",
                f"Overall Rating: {perf_eval.get('overall_rating', 'unknown')}",
                ""
            ])
        
        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                "RECOMMENDATIONS:",
                "-" * 40
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "End of Report"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report_content


# Utility functions for standalone usage
def quick_initialize(force_rebuild: bool = False) -> Dict[str, Any]:
    """Quick initialization for testing"""
    initializer = KnowledgeBaseInitializer()
    return initializer.initialize_and_test(force_rebuild=force_rebuild, run_performance_tests=False)

def full_test_suite(force_rebuild: bool = False) -> Dict[str, Any]:
    """Run full test suite with performance testing"""
    initializer = KnowledgeBaseInitializer()
    return initializer.initialize_and_test(force_rebuild=force_rebuild, run_performance_tests=True)

def health_check() -> Dict[str, Any]:
    """Quick health check of existing knowledge base"""
    try:
        kb = KnowledgeBase()
        if kb.is_ready():
            # Quick test query
            result = kb.query_knowledge_base("test query", include_context=False, max_results=1)
            return {
                "status": "healthy",
                "ready": True,
                "test_query_success": result["status"] == "success",
                "stats": kb.get_knowledge_base_stats()
            }
        else:
            return {
                "status": "not_ready",
                "ready": False,
                "message": "Knowledge base not initialized"
            }
    except Exception as e:
        return {
            "status": "error",
            "ready": False,
            "error": str(e)
        } 