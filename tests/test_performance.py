"""
Performance testing for the EthicalAI API.

This script tests the performance and scalability of the API endpoints
under different load conditions.
"""
import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import aiohttp
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Test configuration
NUM_REQUESTS = 100  # Total number of requests to send
CONCURRENT_REQUESTS = 10  # Number of concurrent requests

# Sample text for testing
SAMPLE_TEXTS = [
    "This is a test sentence about ethics and morality.",
    "Harm reduction and maximizing well-being are key ethical principles.",
    "This text should be evaluated for ethical considerations.",
    "Autonomy and respect for persons is a fundamental ethical principle.",
    "Justice and fairness are important in ethical decision making.",
]


class PerformanceTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.results = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API request and return timing information."""
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                status = response.status
                await response.text()  # Read the response
                end_time = time.time()
                return {
                    "status": status,
                    "duration": end_time - start_time,
                    "success": 200 <= status < 300
                }
        except Exception as e:
            end_time = time.time()
            return {
                "status": str(e),
                "duration": end_time - start_time,
                "success": False
            }

    async def run_load_test(self, endpoint: str, payload: Dict[str, Any], num_requests: int):
        """Run a load test with the given number of concurrent requests."""
        tasks = []
        for _ in range(num_requests):
            task = asyncio.create_task(self.make_request(endpoint, payload))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results.extend([r for r in results if not isinstance(r, Exception)])
        return results

    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics from the test results."""
        if not self.results:
            return {}
        
        durations = [r["duration"] for r in self.results if r.get("success")]
        if not durations:
            return {"success_rate": 0.0}
        
        return {
            "total_requests": len(self.results),
            "successful_requests": len(durations),
            "success_rate": len(durations) / len(self.results),
            "avg_duration": np.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p50": np.percentile(durations, 50),
            "p90": np.percentile(durations, 90),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
        }


@pytest.mark.asyncio
async def test_embed_endpoint_performance(api_client: TestClient):
    """Test the performance of the embed endpoint under load."""
    base_url = "http://test"
    endpoint = "/embed"
    
    # Prepare test data
    test_text = SAMPLE_TEXTS[0]
    payload = {"texts": [test_text]}
    
    async with PerformanceTester(base_url) as tester:
        # Warm-up
        await tester.run_load_test(endpoint, payload, 5)
        
        # Run the actual test
        start_time = time.time()
        await tester.run_load_test(endpoint, payload, NUM_REQUESTS)
        total_time = time.time() - start_time
        
        # Get and log statistics
        stats = tester.get_stats()
        print("\nEmbed Endpoint Performance:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful requests: {stats['successful_requests']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {NUM_REQUESTS/total_time:.2f}")
        print(f"Average latency: {stats['avg_duration']*1000:.2f}ms")
        print(f"p50: {stats['p50']*1000:.2f}ms")
        print(f"p90: {stats['p90']*1000:.2f}ms")
        
        # Assert performance thresholds
        assert stats['success_rate'] >= 0.95, "Success rate below 95%"
        assert stats['p95'] < 1.0, "95th percentile latency too high"


@pytest.mark.asyncio
async def test_analyze_endpoint_performance(api_client: TestClient, tmp_artifacts_dir):
    """Test the performance of the analyze endpoint under load."""
    base_url = "http://test"
    endpoint = "/analyze"
    
    # Create a test axis pack
    axis_pack = {
        "name": "performance_test",
        "axes": [
            {
                "name": "harm_reduction",
                "positive_examples": ["reduces harm", "prevents suffering"],
                "negative_examples": ["causes harm", "inflicts pain"],
                "weight": 1.0
            }
        ]
    }
    axis_pack_path = tmp_artifacts_dir / "performance_axis_pack.json"
    axis_pack_path.write_text(json.dumps(axis_pack))
    
    # Prepare test data
    test_text = SAMPLE_TEXTS[0]
    payload = {
        "texts": [test_text],
        "axis_pack_id": "performance_axis_pack"
    }
    
    async with PerformanceTester(base_url) as tester:
        # Warm-up
        await tester.run_load_test(endpoint, payload, 3)
        
        # Run the actual test
        start_time = time.time()
        await tester.run_load_test(endpoint, payload, NUM_REQUESTS // 2)  # Fewer requests due to higher complexity
        total_time = time.time() - start_time
        
        # Get and log statistics
        stats = tester.get_stats()
        print("\nAnalyze Endpoint Performance:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful requests: {stats['successful_requests']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {(NUM_REQUESTS/2)/total_time:.2f}")
        print(f"Average latency: {stats['avg_duration']*1000:.2f}ms")
        print(f"p50: {stats['p50']*1000:.2f}ms")
        print(f"p90: {stats['p90']*1000:.2f}ms")
        
        # Assert performance thresholds
        assert stats['success_rate'] >= 0.90, "Success rate below 90%"
        assert stats['p95'] < 2.0, "95th percentile latency too high"


def test_concurrent_requests(api_client: TestClient):
    """Test the API's ability to handle concurrent requests."""
    endpoint = "/health"
    url = f"http://test{endpoint}"
    
    def make_request():
        response = api_client.get(endpoint)
        return response.status_code == 200
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(make_request) for _ in range(NUM_REQUESTS)]
        results = [f.result() for f in futures]
    
    success_rate = sum(results) / len(results)
    print(f"\nConcurrent Requests Test ({NUM_REQUESTS} requests, {CONCURRENT_REQUESTS} concurrent):")
    print(f"Success rate: {success_rate:.2%}")
    
    assert success_rate >= 0.95, "Success rate below 95% for concurrent requests"
