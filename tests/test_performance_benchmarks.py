"""Performance benchmarks and stress tests for EthicalAI system.

Tests system performance under various load conditions and measures
response times, throughput, and resource usage.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import statistics
import pytest
from fastapi.testclient import TestClient


class TestPerformanceBenchmarks:
    """Performance benchmarks for EthicalAI system."""
    
    def test_single_embedding_performance(self, api_client: TestClient):
        """Benchmark single text embedding performance."""
        test_text = "AI systems should prioritize human welfare and ethical considerations in all decisions."
        
        # Warm-up request
        api_client.post("/embed", json={"texts": [test_text]})
        
        # Benchmark multiple requests
        response_times = []
        num_requests = 10
        
        for _ in range(num_requests):
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [test_text]})
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Single embedding performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Median: {median_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        
        # Performance assertions (adjust based on system capabilities)
        assert avg_time < 2.0, f"Average response time too high: {avg_time:.3f}s"
        assert max_time < 5.0, f"Maximum response time too high: {max_time:.3f}s"
    
    def test_batch_embedding_performance(self, api_client: TestClient):
        """Benchmark batch embedding performance."""
        test_texts = [
            "AI should respect human autonomy and dignity",
            "Algorithmic bias must be identified and mitigated",
            "Privacy protection is essential in AI systems",
            "Transparency promotes accountability in AI",
            "Fairness should be a core principle in AI design"
        ]
        
        batch_sizes = [1, 5, 10, 20]
        performance_data = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                # Extend test texts by repeating
                extended_texts = (test_texts * ((batch_size // len(test_texts)) + 1))[:batch_size]
            else:
                extended_texts = test_texts[:batch_size]
            
            # Warm-up
            api_client.post("/embed", json={"texts": extended_texts})
            
            # Benchmark
            response_times = []
            num_requests = 5
            
            for _ in range(num_requests):
                start_time = time.time()
                response = api_client.post("/embed", json={"texts": extended_texts})
                end_time = time.time()
                
                assert response.status_code == 200
                response_times.append(end_time - start_time)
            
            avg_time = statistics.mean(response_times)
            throughput = batch_size / avg_time  # texts per second
            
            performance_data[batch_size] = {
                "avg_time": avg_time,
                "throughput": throughput
            }
            
            print(f"Batch size {batch_size}: {avg_time:.3f}s avg, {throughput:.1f} texts/sec")
        
        # Verify throughput scaling
        if len(performance_data) > 1:
            batch_1_throughput = performance_data[1]["throughput"]
            larger_batch_throughput = max(performance_data[bs]["throughput"] 
                                        for bs in performance_data if bs > 1)
            
            # Larger batches should generally have better throughput
            assert larger_batch_throughput >= batch_1_throughput * 0.8, \
                "Batch processing should provide reasonable throughput gains"
    
    def test_concurrent_request_handling(self, api_client: TestClient):
        """Test handling of concurrent requests."""
        test_text = "Ethical AI requires careful consideration of societal impact"
        
        def make_request():
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [test_text]})
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        
        for concurrency in concurrency_levels:
            print(f"Testing concurrency level: {concurrency}")
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                start_time = time.time()
                futures = [executor.submit(make_request) for _ in range(concurrency)]
                results = [future.result() for future in futures]
                total_time = time.time() - start_time
            
            # Analyze results
            successful_requests = sum(1 for r in results if r["success"])
            avg_response_time = statistics.mean(r["response_time"] for r in results)
            total_throughput = successful_requests / total_time
            
            print(f"  Success rate: {successful_requests}/{concurrency}")
            print(f"  Avg response time: {avg_response_time:.3f}s")
            print(f"  Total throughput: {total_throughput:.1f} req/sec")
            
            # All requests should succeed
            assert successful_requests == concurrency, \
                f"Not all concurrent requests succeeded: {successful_requests}/{concurrency}"
            
            # Response times shouldn't degrade too much with concurrency
            assert avg_response_time < 10.0, \
                f"Response time too high under concurrency: {avg_response_time:.3f}s"
    
    def test_large_text_performance(self, api_client: TestClient):
        """Test performance with large text inputs."""
        base_text = "AI ethics involves complex considerations of fairness, transparency, accountability, and human welfare. "
        
        text_sizes = [100, 500, 1000, 2000]  # Number of words
        
        for size in text_sizes:
            # Create text of specified size
            large_text = (base_text * ((size // len(base_text.split())) + 1))
            words = large_text.split()[:size]
            test_text = " ".join(words)
            
            print(f"Testing text size: {len(test_text)} characters, {len(words)} words")
            
            # Measure performance
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [test_text]})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                embedding_dim = len(data["embeddings"][0])
                
                print(f"  Response time: {response_time:.3f}s")
                print(f"  Embedding dimension: {embedding_dim}")
                print(f"  Processing rate: {len(test_text)/response_time:.0f} chars/sec")
                
                # Performance should be reasonable even for large texts
                assert response_time < 30.0, f"Large text processing too slow: {response_time:.3f}s"
                assert embedding_dim > 0, "Should produce valid embeddings"
                
            elif response.status_code in [413, 422]:
                # Text too large - this is acceptable
                print(f"  Text size {size} words rejected (too large)")
            else:
                pytest.fail(f"Unexpected response for text size {size}: {response.status_code}")
    
    def test_memory_usage_stability(self, api_client: TestClient):
        """Test memory usage stability over multiple requests."""
        test_texts = [
            "AI systems must be designed with human values in mind",
            "Ethical considerations should guide AI development",
            "Transparency and accountability are crucial for AI",
            "Fairness and non-discrimination must be ensured",
            "Privacy and data protection are fundamental rights"
        ]
        
        # Make many requests to test for memory leaks
        num_iterations = 50
        response_times = []
        
        for i in range(num_iterations):
            text = test_texts[i % len(test_texts)]
            
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [text]})
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                recent_avg = statistics.mean(response_times[-10:])
                print(f"Iteration {i+1}/{num_iterations}: {recent_avg:.3f}s avg (last 10)")
        
        # Analyze performance stability
        first_quarter = response_times[:num_iterations//4]
        last_quarter = response_times[-num_iterations//4:]
        
        first_avg = statistics.mean(first_quarter)
        last_avg = statistics.mean(last_quarter)
        
        print(f"First quarter average: {first_avg:.3f}s")
        print(f"Last quarter average: {last_avg:.3f}s")
        print(f"Performance degradation: {((last_avg/first_avg - 1) * 100):.1f}%")
        
        # Performance shouldn't degrade significantly over time
        degradation_ratio = last_avg / first_avg
        assert degradation_ratio < 2.0, \
            f"Significant performance degradation detected: {degradation_ratio:.2f}x slower"
    
    def test_error_recovery_performance(self, api_client: TestClient):
        """Test performance after error conditions."""
        valid_text = "AI should promote human flourishing and wellbeing"
        
        # First, cause some errors
        error_requests = [
            {"texts": []},  # Empty array
            {"texts": [""]},  # Empty string
            {"invalid": "data"},  # Invalid format
        ]
        
        for error_req in error_requests:
            response = api_client.post("/embed", json=error_req)
            # Should get error response
            assert response.status_code in [400, 422]
        
        # Now test that valid requests still work efficiently
        recovery_times = []
        num_recovery_tests = 10
        
        for _ in range(num_recovery_tests):
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [valid_text]})
            end_time = time.time()
            
            assert response.status_code == 200
            recovery_times.append(end_time - start_time)
        
        avg_recovery_time = statistics.mean(recovery_times)
        print(f"Average recovery time after errors: {avg_recovery_time:.3f}s")
        
        # Should recover quickly from errors
        assert avg_recovery_time < 5.0, \
            f"Slow recovery after errors: {avg_recovery_time:.3f}s"


class TestStressTests:
    """Stress tests for system limits and robustness."""
    
    def test_rapid_fire_requests(self, api_client: TestClient):
        """Test rapid succession of requests."""
        test_text = "Rapid fire test for AI ethics evaluation"
        
        num_requests = 20
        max_interval = 0.1  # 100ms between requests
        
        response_times = []
        success_count = 0
        
        for i in range(num_requests):
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [test_text]})
            end_time = time.time()
            
            response_times.append(end_time - start_time)
            
            if response.status_code == 200:
                success_count += 1
            
            # Small delay between requests
            if i < num_requests - 1:
                time.sleep(max_interval)
        
        success_rate = success_count / num_requests
        avg_response_time = statistics.mean(response_times)
        
        print(f"Rapid fire test results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average response time: {avg_response_time:.3f}s")
        
        # Should handle rapid requests reasonably well
        assert success_rate >= 0.8, f"Low success rate in rapid fire test: {success_rate:.1%}"
        assert avg_response_time < 10.0, f"High response time in rapid fire test: {avg_response_time:.3f}s"
    
    def test_edge_case_inputs(self, api_client: TestClient):
        """Test performance with edge case inputs."""
        edge_cases = [
            "a",  # Single character
            "AI",  # Very short
            "   ",  # Whitespace only
            "🤖🧠💭",  # Unicode/emoji only
            "AI " * 100,  # Repetitive text
            "The quick brown fox jumps over the lazy dog. " * 50,  # Long repetitive
        ]
        
        for i, text in enumerate(edge_cases):
            # Make preview printing ASCII-safe to avoid UnicodeEncodeError on some consoles (e.g., Windows cp1252)
            preview = text[:50]
            safe_preview = preview.encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Testing edge case {i+1}: '{safe_preview}{'...' if len(text) > 50 else ''}'")
            
            start_time = time.time()
            response = api_client.post("/embed", json={"texts": [text]})
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should handle edge cases gracefully
            assert response.status_code in [200, 400, 422], \
                f"Unexpected status for edge case: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "embeddings" in data
                assert len(data["embeddings"]) == 1
                print(f"  Success: {response_time:.3f}s")
            else:
                print(f"  Rejected (status {response.status_code}): {response_time:.3f}s")
            
            # Should respond quickly even for edge cases
            assert response_time < 5.0, f"Slow response for edge case: {response_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
