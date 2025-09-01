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
from pathlib import Path

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
    def __init__(self, client: TestClient):
        self.client = client
        self.results: List[Dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API request and return timing information using TestClient in a worker thread."""
        start_time = time.time()
        try:
            response = await asyncio.to_thread(self.client.post, endpoint, json=payload)
            status = response.status_code
            # consume content to mimic network read cost (optional)
            _ = response.text
            end_time = time.time()
            return {
                "status": status,
                "duration": end_time - start_time,
                "success": 200 <= status < 300,
            }
        except Exception as e:
            end_time = time.time()
            return {"status": str(e), "duration": end_time - start_time, "success": False}

    async def run_load_test(self, endpoint: str, payload: Dict[str, Any], num_requests: int, concurrency: int = CONCURRENT_REQUESTS):
        """Run a load test with bounded concurrency (default CONCURRENT_REQUESTS)."""
        sem = asyncio.Semaphore(concurrency)

        async def worker() -> Dict[str, Any]:
            await sem.acquire()
            try:
                return await self.make_request(endpoint, payload)
            finally:
                sem.release()

        tasks = [asyncio.create_task(worker()) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results.extend([r for r in results if not isinstance(r, Exception)])
        return results

    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics from the test results."""
        if not self.results:
            return {}
        
        durations = [r["duration"] for r in self.results if r.get("success")]
        if not durations:
            # Return minimal stats to avoid KeyError in callers
            return {
                "total_requests": len(self.results),
                "successful_requests": 0,
                "success_rate": 0.0,
            }
        
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
async def test_embed_endpoint_performance(api_client_real_encoder: TestClient):
    """Test the performance of the embed endpoint under load."""
    endpoint = "/embed"
    
    # Prepare test data
    test_text = SAMPLE_TEXTS[0]
    payload = {"texts": [test_text]}
    
    async with PerformanceTester(api_client_real_encoder) as tester:
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
        if 'avg_duration' in stats:
            print(f"Average latency: {stats['avg_duration']*1000:.2f}ms")
            print(f"p50: {stats['p50']*1000:.2f}ms")
            print(f"p90: {stats['p90']*1000:.2f}ms")
        
        # Assert performance thresholds
        assert stats['success_rate'] >= 0.95, "Success rate below 95%"
        if 'p95' in stats:
            assert stats['p95'] < 1.5, "95th percentile latency too high"


def _ensure_axis_pack(axis_pack_id: str, d: int = 768, k: int = 2) -> Path:
    """Create a minimal compatible axis pack under data/axes/ if missing."""
    axes_dir = Path("data") / "axes"
    axes_dir.mkdir(parents=True, exist_ok=True)
    pack_path = axes_dir / f"{axis_pack_id}.json"
    if not pack_path.exists():
        names = [f"axis_{i}" for i in range(k)]
        Q = [[1.0 if (i < k and j == i) else 0.0 for j in range(k)] for i in range(d)]
        pack = {
            "names": names,
            "Q": Q,
            "lambda": [1.0] * k,
            "beta": [0.0] * k,
            "weights": [1.0 / k] * k,
            "mu": {},
            "meta": {"id": axis_pack_id, "test": True},
        }
        pack_path.write_text(json.dumps(pack), encoding="utf-8")
    return pack_path


@pytest.mark.asyncio
async def test_analyze_endpoint_performance(api_client_real_encoder: TestClient, tmp_artifacts_dir):
    """Test the performance of the analyze endpoint under load."""
    endpoint = "/analyze"
    
    # Ensure a compatible axis pack exists under data/axes/
    axis_pack_id = "performance_axis_pack"
    _ensure_axis_pack(axis_pack_id)
    
    # Prepare test data
    test_text = SAMPLE_TEXTS[0]
    payload = {
        "texts": [test_text],
        "axis_pack_id": axis_pack_id,
    }
    
    async with PerformanceTester(api_client_real_encoder) as tester:
        # Warm-up
        await tester.run_load_test(endpoint, payload, 3, concurrency=3)
        
        # Run the actual test
        start_time = time.time()
        await tester.run_load_test(endpoint, payload, NUM_REQUESTS // 2, concurrency=5)  # Fewer requests and bounded concurrency due to higher complexity
        total_time = time.time() - start_time
        
        # Get and log statistics
        stats = tester.get_stats()
        print("\nAnalyze Endpoint Performance:")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful requests: {stats['successful_requests']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests per second: {(NUM_REQUESTS/2)/total_time:.2f}")
        if 'avg_duration' in stats:
            print(f"Average latency: {stats['avg_duration']*1000:.2f}ms")
            print(f"p50: {stats['p50']*1000:.2f}ms")
            print(f"p90: {stats['p90']*1000:.2f}ms")
        
        # Assert performance thresholds
        assert stats['success_rate'] >= 0.90, "Success rate below 90%"
        if 'p95' in stats:
            assert stats['p95'] < 2.6, "95th percentile latency too high"


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
