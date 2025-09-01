# EthicalAI Testing Guide

## Overview

This guide provides comprehensive documentation for the EthicalAI test suite, including unit tests, integration tests, performance benchmarks, and end-to-end tests. The testing strategy emphasizes real-world validation through the use of actual encoder models in testing environments.

## Test Categories

### 1. Unit Tests
- **Purpose**: Validate individual components in isolation
- **Location**: `tests/unit/`
- **Key Features**:
  - Mock external dependencies
  - Fast execution
  - High test coverage
  - Isolated component testing

### 2. Integration Tests
- **Purpose**: Verify component interactions
- **Location**: `tests/integration/`
- **Key Features**:
  - Test component integration
  - Verify API contracts
  - Validate data flow between components
  - Includes both happy paths and error cases

### 3. Performance Tests
- **Purpose**: Measure and validate system performance
- **Location**: `tests/performance/`
- **Key Features**:
  - Benchmark critical paths
  - Measure response times under load
  - Validate memory usage patterns
  - Test with real encoder models

### 4. End-to-End Tests
- **Purpose**: Validate complete user workflows
- **Location**: `tests/e2e/`
- **Key Features**:
  - Full stack testing
  - Real browser automation
  - User journey validation
  - Cross-browser compatibility

## Real Encoder Testing

The test suite includes the capability to test with real encoder models by setting the `COHERENCE_TEST_REAL_ENCODER=1` environment variable. This is particularly useful for:

1. **Model Validation**: Ensuring the encoder produces expected embeddings
2. **Performance Benchmarking**: Measuring real-world performance characteristics
3. **Integration Testing**: Validating the complete pipeline with actual models

### Enabling Real Encoder Tests

```bash
# Run all tests with real encoder
COHERENCE_TEST_REAL_ENCODER=1 pytest

# Run specific test with real encoder
COHERENCE_TEST_REAL_ENCODER=1 pytest tests/test_encoder.py -v
```

### Performance Benchmarks

Performance tests include:

1. **Single Embedding Performance**
   - Measures latency for individual text embeddings
   - Tracks memory usage per request
   - Validates response consistency

2. **Batch Processing**
   - Tests with various batch sizes
   - Measures throughput (embeddings/second)
   - Validates memory efficiency

3. **Concurrent Requests**
   - Tests system under concurrent load
   - Measures throughput and error rates
   - Validates thread safety

4. **Long-Running Tests**
   - Validates memory stability over time
   - Detects resource leaks
   - Ensures consistent performance

## Test Environment

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend tests)
- Dependencies from `requirements.txt` and `requirements-test.txt`
- Sufficient disk space for test artifacts

### Configuration

Environment variables for test configuration:

```bash
# Required for real encoder tests
COHERENCE_TEST_REAL_ENCODER=1

# Directory for test artifacts (default: ./test_artifacts)
COHERENCE_ARTIFACTS_DIR=./test_artifacts

# Log level (debug, info, warning, error, critical)
COHERENCE_LOG_LEVEL=info

# Override default encoder model
COHERENCE_ENCODER=all-mpnet-base-v2
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing
```

### Test Selection

Run specific test categories:

```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance benchmarks
pytest tests/performance/ -m "performance"

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Frontend Tests

```bash
# Navigate to UI directory
cd ui

# Install dependencies
npm install

# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## Test Configuration

### Pytest Configuration

The project includes a `pytest.ini` file with default configurations:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --cov=src --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    performance: performance benchmark tests
    integration: integration tests
    e2e: end-to-end tests
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COHERENCE_TEST_REAL_ENCODER` | Use real encoder model (1) or mocks (0) | 0 |
| `COHERENCE_ARTIFACTS_DIR` | Directory for test artifacts | `./test_artifacts` |
| `COHERENCE_LOG_LEVEL` | Logging level | `info` |
| `COHERENCE_ENCODER` | Override default encoder model | `all-mpnet-base-v2` |

## Test Fixtures

The test suite includes several fixtures to support testing:

### Backend Fixtures (`tests/conftest.py`)

```python
@pytest.fixture(scope="session")
def tmp_artifacts_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="module")
def api_client(tmp_artifacts_dir):
    """Create a test client with overridden settings."""
    settings.ARTIFACTS_DIR = tmp_artifacts_dir
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def real_encoder():
    """Fixture that provides a real encoder model."""
    if not os.environ.get("COHERENCE_TEST_REAL_ENCODER"):
        pytest.skip("Real encoder tests disabled. Set COHERENCE_TEST_REAL_ENCODER=1 to enable.")
    return get_default_encoder()
```

### Frontend Mocks (`ui/src/__mocks__/server.ts`)

```typescript
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
```

## Writing Tests

### Backend Test Example

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
class TestEmbeddingAPI:
    def test_text_embedding(self, api_client):
        """Test that text embedding returns expected vector dimensions."""
        response = api_client.post("/v1/embed", json={"text": "Test input"})
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 768  # Expected embedding dimension

    @pytest.mark.performance
    def test_embedding_performance(self, api_client, benchmark):
        """Benchmark embedding performance."""
        def run():
            return api_client.post("/v1/embed", json={"text": "Performance test"})
        
        # Run benchmark
        result = benchmark(run)
        assert result.status_code == 200
```

### Frontend Test Example

```typescript
// ui/src/components/__tests__/Analyzer.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Analyzer } from '../Analyzer'

describe('Analyzer', () => {
  it('processes text input and displays results', async () => {
    render(<Analyzer />)
    
    // Simulate user input
    const input = screen.getByRole('textbox')
    await userEvent.type(input, 'Test input')
    
    // Click analyze button
    const button = screen.getByRole('button', { name: /analyze/i })
    await userEvent.click(button)
    
    // Verify loading state
    expect(screen.getByText(/analyzing/i)).toBeInTheDocument()
    
    // Wait for results
    await waitFor(() => {
      expect(screen.getByText(/results/i)).toBeInTheDocument()
    })
  })
})
```

## Test Categories in Detail

### 1. Unit Tests

#### Encoder Tests
- Model loading and initialization
- Text preprocessing
- Embedding generation
- Batch processing
- Error handling

#### Axis Pack Tests
- Pack loading and validation
- Vector operations
- Similarity calculations
- Threshold application
- Metadata handling

### 2. Integration Tests

#### API Endpoint Tests
- Request validation
- Response formatting
- Error conditions
- Authentication/authorization
- Rate limiting

#### Pipeline Tests
- End-to-end text processing
- Multi-stage transformations
- Error propagation
- Resource cleanup

### 3. Performance Tests

#### Load Testing
- Concurrent user simulation
- Throughput measurement
- Resource utilization
- Scaling behavior

#### Stress Testing
- System limits
- Failure recovery
- Degradation patterns
- Memory management

### 4. End-to-End Tests

#### User Flows
- Complete analysis workflow
- Error scenarios
- Edge cases
- Cross-browser compatibility

#### API Contract Tests
- Request/response validation
- Version compatibility
- Backward compatibility
- Documentation accuracy

## Best Practices

### Writing Maintainable Tests

1. **Descriptive Test Names**
   ```python
   # Bad
   def test_case1():
   
   # Good
   def test_embedding_returns_expected_dimensions():
   ```

2. **Use Fixtures for Setup**
   ```python
   @pytest.fixture
def sample_texts():
    return ["First text", "Second text"]

   def test_batch_processing(sample_texts):
       # Test code here
   ```

3. **Assertion Clarity**
   ```python
   # Less clear
   assert result == expected
   
   # More descriptive
   assert result["status"] == "success", "Expected successful status"
   assert len(result["embedding"]) == 768, "Embedding dimension mismatch"
   ```

### Performance Testing Guidelines

1. **Baseline Establishment**
   - Establish performance baselines
   - Document expected performance characteristics
   - Set performance budgets

2. **Continuous Monitoring**
   - Track performance metrics over time
   - Set up alerts for regressions
   - Document performance trends

3. **Realistic Testing**
   - Use production-like data
   - Test with realistic load patterns
   - Consider network conditions

## Troubleshooting

### Common Issues

1. **Tests Hanging**
   - Check for unclosed resources
   - Verify timeouts are appropriate
   - Look for deadlocks in concurrent code

2. **Intermittent Failures**
   - Check for race conditions
   - Ensure proper test isolation
   - Verify test data consistency

3. **Performance Regressions**
   - Check for new dependencies
   - Review recent code changes
   - Verify resource constraints

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pytest's debugging features
pytest --pdb  # Drop into debugger on failure
pytest --trace  # Start debugger immediately
```

## CI/CD Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      env:
        COHERENCE_TEST_REAL_ENCODER: 1
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

## License

This testing guide and associated test suite are part of the EthicalAI project and are made available under the same license as the main project.
- Error handling and recovery
- Performance under load
- Accessibility compliance

### 4. Performance Benchmarks (`test_performance_benchmarks.py`)

#### Response Time Testing
- Single embedding: < 2.0s average
- Batch processing: Scalable throughput
- Concurrent requests: Stable under load
- Large text inputs: < 30.0s processing

#### Stress Testing
- Rapid-fire requests (20 req/100ms)
- Memory leak detection (50+ iterations)
- Error recovery performance
- Edge case input handling

## 🎯 Expected Test Results

### Success Criteria

#### Backend Tests
- All API endpoints respond correctly
- Embedding generation produces valid vectors
- Ethical analysis provides meaningful scores
- Error handling is robust and informative

#### Frontend Tests  
- Components render without errors
- API integration works seamlessly
- User interactions trigger correct behaviors
- Error states are handled gracefully

#### Performance Tests
- Response times within acceptable ranges
- System remains stable under load
- Memory usage doesn't grow over time
- Concurrent requests handled properly

### Performance Benchmarks
- **Single Embedding**: < 2.0s average response time
- **Batch Processing**: Linear scaling with batch size
- **Concurrent Requests**: 80%+ success rate under load
- **Large Text Processing**: < 30.0s for 2000+ words

## 🐛 Troubleshooting

### Common Issues

#### Backend Test Issues
```bash
# Model loading timeout
export COHERENCE_TEST_MODE=true
export COHERENCE_ENCODER=all-mpnet-base-v2

# Fixture scope errors
# Fixed in conftest.py with proper scope alignment

# Import errors
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src
```

#### Frontend Test Issues
```bash
# MSW version compatibility
npm install msw@^2.0.0

# Missing test dependencies
npm install @testing-library/react @testing-library/jest-dom vitest jsdom

# TypeScript errors
# Ensure proper type definitions in setupTests.ts
```

#### Performance Test Issues
```bash
# Slow test execution
# Reduce iteration counts in performance tests
# Use smaller test datasets

# Memory issues
# Monitor system resources during tests
# Adjust batch sizes if needed
```

### Debug Mode
```bash
# Verbose backend testing
python -m pytest tests/ -v -s --tb=long

# Frontend test debugging
cd ui && npm test -- --reporter=verbose

# Performance profiling
python -m pytest tests/test_performance_benchmarks.py -v -s --profile
```

## 📈 Continuous Integration

### GitHub Actions Workflow
```yaml
name: Comprehensive Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run backend tests
        run: python run_comprehensive_tests.py
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install frontend dependencies
        run: cd ui && npm install
      - name: Run frontend tests
        run: cd ui && npm test
```

## 🎉 Test Execution Summary

When all tests pass, you should see:

```
📊 COMPREHENSIVE TEST RESULTS SUMMARY
====================================

🔧 Backend API Tests:
  tests/test_comprehensive_e2e.py: ✅ PASS (45.2s)
  tests/test_axis_packs_comprehensive.py: ✅ PASS (32.1s)
  tests/test_frontend_integration.py: ✅ PASS (28.7s)
  tests/test_performance_benchmarks.py: ✅ PASS (67.3s)

🎨 Frontend Tests:
  Frontend tests: ✅ PASS

📈 OVERALL RESULTS:
  Total Tests: 5
  Passed: 5
  Failed: 0
  Success Rate: 100.0%
  Total Duration: 173.3s

🎯 TEST COVERAGE AREAS:
  ✅ Text Embedding & Vector Generation
  ✅ Ethical Evaluation Pipeline
  ✅ Axis Pack Loading & Configuration
  ✅ Vector Topology Analysis
  ✅ Batch Processing
  ✅ What-if Analysis
  ✅ Frontend-Backend Integration
  ✅ Error Handling & Edge Cases
  ✅ Performance Benchmarks
  ✅ API Contract Compliance

🎉 ALL TESTS PASSED!
  • EthicalAI system is functioning correctly
  • All core functionality is working
  • Performance is within acceptable ranges
  • Frontend-backend integration is solid
```

This comprehensive test suite ensures that the EthicalAI system functions correctly across all components, from low-level vector operations to high-level ethical analysis and user interface integration.
