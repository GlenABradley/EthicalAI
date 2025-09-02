# EthicalAI Testing Guide

## Overview

This guide provides comprehensive documentation for the EthicalAI test suite. The testing strategy emphasizes **real-world validation** through the use of actual encoder models - **no mocks or stubs** are used in the test suite.

## Core Testing Philosophy

### Real Encoder Testing

**IMPORTANT**: All tests use the real SentenceTransformer encoder model. This is enforced by:

- Setting `COHERENCE_TEST_REAL_ENCODER=1` in `tests/conftest.py`
- Preloading the encoder once per session for efficiency
- No mocks or stubs are used in the test suite

## Test Structure

### Test Files Organization

```text
tests/
├── api/                        # API endpoint tests
│   ├── test_frames_endpoints.py
│   ├── test_frames_index_contracts.py
│   ├── test_frames_stats.py
│   └── test_router.py
├── memory/                     # Memory store tests  
│   └── test_store_stubs.py
├── conftest.py                 # Global fixtures and configuration
├── test_advanced_axis_integration.py
├── test_axis_builder.py
├── test_axis_packs.py
├── test_axis_packs_comprehensive.py
├── test_comprehensive_e2e.py
├── test_encoder.py
├── test_ethicalai_integration.py
├── test_frames.py
├── test_frames_contracts.py
├── test_frontend_integration.py
├── test_integration.py
├── test_models.py
├── test_performance_benchmarks.py
├── test_pipeline.py
├── test_resonance_384d.py
└── test_vector_topology.py
```

## Test Categories

### 1. Core Component Tests

#### Encoder Tests (`test_encoder.py`)

- Tests SentenceTransformer model loading and initialization
- Validates embedding generation (384-dimensional vectors)
- Verifies batch processing capabilities
- Tests text preprocessing and tokenization

#### Axis Pack Tests (`test_axis_packs.py`, `test_axis_packs_comprehensive.py`)

- Validates axis pack loading from NPZ files
- Tests projection matrix operations (Q matrix)
- Verifies threshold calibration
- Tests axis metadata handling

#### Pipeline Tests (`test_pipeline.py`)

- End-to-end text processing pipeline
- Multi-stage transformations
- Error propagation and handling
- Resource cleanup validation

### 2. Integration Tests

#### API Integration (`test_integration.py`, `test_ethicalai_integration.py`)

- Complete API endpoint testing
- Request/response validation
- Error handling scenarios
- Cross-component data flow

#### Frame Management (`test_frames.py`, `test_frames_contracts.py`)

- SQLite database operations
- Frame storage and retrieval
- Semantic memory persistence
- Index building and search

#### Advanced Axis Integration (`test_advanced_axis_integration.py`)

- Complex axis pack operations
- Multi-dimensional projections
- Choquet integral calculations
- Veto span detection

### 3. Performance Tests (`test_performance_benchmarks.py`)

#### Response Time Benchmarks

- **Single Embedding**: Target < 2.0s average response time
- **Batch Processing**: Linear scaling with batch size
- **Large Text Inputs**: < 30.0s for 2000+ word documents
- **Concurrent Requests**: 80%+ success rate under load

#### Stress Testing

- Rapid-fire requests (20 requests/100ms)
- Memory leak detection (50+ iterations)
- Error recovery performance
- Edge case input handling

#### Resource Monitoring

- CPU utilization tracking
- Memory usage patterns
- Thread pool efficiency
- Database connection pooling

### 4. End-to-End Tests (`test_comprehensive_e2e.py`)

#### Complete Workflows

- Text analysis pipeline from input to output
- Ethical evaluation with decision proofs
- Multi-axis scoring and aggregation
- Frame storage and retrieval cycles

#### Frontend Integration (`test_frontend_integration.py`)

- API contract validation
- Component rendering tests
- User interaction flows
- Error state handling

## Test Environment Setup

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Node.js 18+ (for frontend tests)
- Sufficient RAM for model loading (4GB minimum)
- Windows/Linux/macOS support

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Windows-specific dependencies
pip install -r requirements-windows.txt

# Frontend dependencies (optional)
cd ui && npm install
```

### Configuration

The test suite is configured via `tests/conftest.py`:

```python
# Enforced settings in conftest.py
os.environ["COHERENCE_TEST_REAL_ENCODER"] = "1"  # Always use real encoder
os.environ["COHERENCE_TEST_MODE"] = "true"        # Enable test mode

# Optional environment variables
COHERENCE_ARTIFACTS_DIR=./test_artifacts  # Test artifact storage
COHERENCE_LOG_LEVEL=info                  # Logging verbosity
```

## Running Tests

### Quick Start

```bash
# Run all tests (uses real encoder automatically)
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Windows-Specific Commands

```powershell
# PowerShell script
.\run_tests.ps1

# Batch file
.\run_tests.bat

# Direct batch execution
.\run_tests_direct.bat
```

### Selective Test Execution

```bash
# Run specific test file
pytest tests/test_encoder.py -v

# Run API tests only
pytest tests/api/ -v

# Run performance benchmarks
pytest tests/test_performance_benchmarks.py -v

# Run comprehensive E2E tests
pytest tests/test_comprehensive_e2e.py -v

# Skip slow tests
pytest -m "not slow"
```

### Frontend Tests

```bash
# Navigate to UI directory
cd ui

# Install dependencies
npm install

# Run Vitest suite
npm test

# Run with coverage
npm run test:coverage

# Run Playwright E2E tests
npm run test:e2e

# Run in watch mode
npm run test:watch
```

## Test Configuration Files

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra --strict-markers
markers =
    slow: marks tests as slow
    performance: performance benchmark tests
    integration: integration tests
    e2e: end-to-end tests
```

### mypy.ini

```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
```

### ruff.toml

```toml
target-version = "py310"
line-length = 120
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]  # Line too long
```

## Key Test Fixtures

### Global Fixtures (`tests/conftest.py`)

```python
@pytest.fixture(scope="session")
def encoder():
    """Preloaded real encoder for all tests."""
    # Real encoder is loaded once per session
    from coherence.encoder import CoherenceEncoder
    return CoherenceEncoder()

@pytest.fixture(scope="session")
def axis_pack():
    """Default axis pack for testing."""
    from coherence.axis.loader import load_axis_pack
    return load_axis_pack("ap_1756646151")

@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)

@pytest.fixture
def api_client_real_encoder(api_client, encoder):
    """API client with real encoder preloaded."""
    # Ensures encoder is loaded before API calls
    return api_client
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

## Expected Test Results

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

## Troubleshooting Guide

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

## Continuous Integration

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

## Test Execution Summary

When all tests pass, you should see:

```text
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

## Conclusion

The EthicalAI test suite provides comprehensive validation of the system through:

- **Real encoder testing** - No mocks or stubs, ensuring real-world accuracy
- **Multi-level coverage** - From unit tests to end-to-end workflows
- **Performance benchmarking** - Validated response times and resource usage
- **Integration testing** - Full API and component interaction validation

All tests use the actual SentenceTransformer model to ensure that test results accurately reflect production behavior. This approach provides high confidence in the system's reliability and performance characteristics.
