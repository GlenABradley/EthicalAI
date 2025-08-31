# EthicalAI Comprehensive Testing Guide

This guide covers the complete test suite for the EthicalAI project, including backend API tests, ethical evaluation tests, frontend integration tests, and performance benchmarks.

## 🎯 Test Coverage Overview

The comprehensive test suite covers:

### Backend API Testing
- ✅ **Text Embedding & Vector Generation**
- ✅ **Ethical Evaluation Pipeline**
- ✅ **Axis Pack Loading & Configuration**
- ✅ **Vector Topology Analysis**
- ✅ **Batch Processing Capabilities**
- ✅ **What-if Analysis Functionality**
- ✅ **Error Handling & Edge Cases**
- ✅ **API Contract Compliance**

### Frontend Integration Testing
- ✅ **React Component Integration**
- ✅ **Frontend-Backend API Communication**
- ✅ **Real-time Analysis Workflow**
- ✅ **Visualization Data Flow**
- ✅ **Error Handling & User Experience**
- ✅ **Performance & Responsiveness**
- ✅ **Accessibility Compliance**

### Ethical Evaluation Testing
- ✅ **All Axis Packs (Consequentialism, Deontology, Virtue, etc.)**
- ✅ **Cross-Axis Ethical Analysis**
- ✅ **Real-world Ethical Scenarios**
- ✅ **Ethical Consistency Validation**
- ✅ **Complex Moral Dilemma Analysis**

### Performance & Stress Testing
- ✅ **Single & Batch Embedding Performance**
- ✅ **Concurrent Request Handling**
- ✅ **Large Text Input Processing**
- ✅ **Memory Usage Stability**
- ✅ **Error Recovery Performance**
- ✅ **Rapid-fire Request Testing**

## 🚀 Quick Start - Run All Tests

### Option 1: Automated Test Runner (Recommended)
```bash
# Run the comprehensive test suite
python run_comprehensive_tests.py
```

This will execute all test categories and provide a detailed summary report.

### Option 2: Manual Test Execution

#### Backend Tests
```bash
# Core end-to-end tests
python -m pytest tests/test_end_to_end.py -v

# Comprehensive API tests
python -m pytest tests/test_comprehensive_e2e.py -v

# Frontend integration tests (backend perspective)
python -m pytest tests/test_frontend_integration.py -v

# Axis pack and ethical evaluation tests
python -m pytest tests/test_axis_packs_comprehensive.py -v

# Performance benchmarks
python -m pytest tests/test_performance_benchmarks.py -v -s
```

#### Frontend Tests
```bash
cd ui
npm test
```

## 📋 Prerequisites

### Backend Testing Requirements
```bash
# Install Python dependencies
pip install -r requirements.txt

# Additional test dependencies
pip install pytest pytest-asyncio httpx numpy scikit-learn
```

### Frontend Testing Requirements
```bash
cd ui
npm install
```

Required packages:
- `@testing-library/react`
- `@testing-library/jest-dom`
- `vitest`
- `msw` (v2.x)
- `jsdom`

## 🔧 Test Configuration

### Backend Test Configuration

#### Pytest Configuration (`pytest.ini`)
```ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
addopts = -v --tb=short --cov=src --cov-report=term-missing
python_files = test_*.py
python_functions = test_*
log_cli = true
log_cli_level = INFO
timeout = 300
```

#### Test Fixtures (`tests/conftest.py`)
- **Mocked encoder**: Prevents model downloading during tests
- **Temporary artifacts directory**: Isolated test environment
- **API client fixture**: Configured TestClient with proper scoping

### Frontend Test Configuration

#### Vitest Configuration (`ui/vitest.config.ts`)
```typescript
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
})
```

#### MSW Setup (`ui/src/setupTests.ts`)
```typescript
import { beforeAll, afterEach, afterAll } from 'vitest'
import { server } from './__mocks__/server'

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

## 📊 Test Categories Detailed

### 1. Core Backend API Tests (`test_comprehensive_e2e.py`)

#### Text Embedding Tests
- Single text embedding validation
- Batch embedding processing
- Vector dimensionality verification
- Embedding uniqueness validation

#### Ethical Evaluation Tests
- Ethical vs harmful text analysis
- Complex ethical scenario processing
- Multi-axis ethical evaluation
- Detailed analysis generation

#### Vector Topology Tests
- Similarity matrix generation
- Clustering analysis
- Topology property validation
- High-dimensional vector operations

### 2. Axis Pack Tests (`test_axis_packs_comprehensive.py`)

#### Individual Axis Pack Testing
- **Consequentialism**: Outcome-based ethical evaluation
- **Deontology**: Duty-based ethical evaluation  
- **Virtue Ethics**: Character-based ethical evaluation
- **Intent-based**: Good/bad intention analysis

#### Cross-Axis Analysis
- Multi-framework ethical evaluation
- Consistency checking across axes
- Complex moral dilemma analysis
- Real-world scenario testing

### 3. Frontend Integration Tests (`comprehensive-integration.test.tsx`)

#### Component Integration
- App component rendering
- Text input and analysis workflow
- Real-time feedback systems
- Visualization component integration

#### API Integration
- Health check monitoring
- Embedding request handling
- Analysis result processing
- Error state management

#### User Experience Testing
- Loading states and feedback
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
