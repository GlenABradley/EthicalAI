# EthicalAI Architecture

## System Overview

EthicalAI is a comprehensive framework for analyzing and evaluating text content against ethical dimensions. The system combines natural language processing with ethical reasoning to provide insights and decision support for content evaluation.

## Core Components

### 1. Coherence Engine (`coherence/`)
The foundation layer responsible for text processing and embeddings:

- **Embeddings**: Utilizes SentenceTransformer models (default: `all-mpnet-base-v2`) to generate dense vector representations of text
- **Pipeline**: Manages the end-to-end processing of text through the analysis pipeline
- **FastAPI App**: Provides the web interface and API endpoints for system interaction
- **Artifact Management**: Handles persistence of axis packs, models, and analysis results

### 2. EthicalAI Integration (`ethicalai/`)
Implements the ethical reasoning and evaluation capabilities:

- **Axes**: Defines and manages ethical dimensions for analysis
- **Evaluator**: Core logic for evaluating content against ethical dimensions
- **Constitution**: Reranks content based on ethical principles
- **Interaction**: Policy middleware for request/response handling

## Data Flow

1. **Initialization**
   - Load or build axis packs containing ethical dimensions
   - Initialize embedding model and processing pipeline
   - Configure evaluation thresholds and parameters

2. **Text Processing**
   - Input text is received via API or UI
   - Text is tokenized and processed through the embedding model
   - Dense vector representations are generated for analysis

3. **Ethical Evaluation**
   - Text embeddings are compared against axis vectors
   - Scores are calculated for each ethical dimension
   - Results are aggregated and formatted for presentation

4. **Response Generation**
   - Evaluation results are compiled into a structured response
   - Supporting evidence and explanations are included
   - Alternative phrasings may be suggested when applicable

## API Endpoints

### Core Endpoints
- `GET /health/ready` - System health check
- `GET /health/live` - Liveness probe
- `GET /v1/axes` - List available axis packs
- `POST /v1/axes/build` - Build a new axis pack
- `POST /v1/axes/activate` - Activate a specific axis pack
- `GET /v1/axes/active` - Get currently active axis pack

### Evaluation Endpoints
- `POST /v1/eval/text` - Evaluate text against active axes
- `POST /v1/eval/batch` - Batch evaluate multiple texts
- `GET /v1/eval/thresholds` - Get current evaluation thresholds

### Advanced Features
- `POST /v1/constitution/rank` - Rerank content based on ethical principles
- `POST /v1/interaction/respond` - Generate ethical responses to queries

## Data Artifacts

### Axis Packs
- `artifacts/axis_pack:<id>.npz` - Binary file containing axis vectors
- `artifacts/axis_pack:<id>.meta.json` - Metadata including axis definitions and thresholds

### Calibration Reports
- `reports/calibration:<id>/` - Directory containing calibration results
  - `summary.json` - Summary statistics and metrics
  - `calibration_data.csv` - Raw calibration data
  - `performance_metrics.json` - Performance metrics for the axis pack

## Configuration

### Environment Variables
- `COHERENCE_ARTIFACTS_DIR`: Directory for storing artifacts (default: `./artifacts`)
- `COHERENCE_ENCODER`: Override default sentence transformer model
- `COHERENCE_LOG_LEVEL`: Logging verbosity (debug, info, warning, error, critical)
- `COHERENCE_TEST_REAL_ENCODER`: Set to 1 to use real encoder in tests

### Configuration Files
- `configs/app.yaml` - Application configuration
- `configs/logging.yaml` - Logging configuration
- `configs/axis_packs/*.json` - Predefined axis pack configurations

## Performance Considerations

- **Model Loading**: The sentence transformer model is loaded once at startup to minimize latency
- **Batch Processing**: Supports batch processing of multiple texts for improved throughput
- **Caching**: Implements caching of intermediate results where appropriate
- **Memory Management**: Includes safeguards against memory exhaustion with large inputs

## Security

- Input validation on all API endpoints
- Rate limiting to prevent abuse
- Secure handling of sensitive data in memory
- Comprehensive error handling and logging

## Monitoring and Observability

- Health check endpoints for monitoring
- Structured logging for operational insights
- Performance metrics collection
- Error tracking and alerting

## Testing Strategy

- Unit tests for individual components
- Integration tests for API endpoints
- Performance benchmarks for critical paths
- Real-encoder testing with `COHERENCE_TEST_REAL_ENCODER=1`
- End-to-end tests for complete workflows

## Deployment

### Requirements
- Python 3.8+
- Dependencies from `requirements.txt`
- Sufficient disk space for models and artifacts

### Deployment Options
1. **Local Development**:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

2. **Production**:
   - Use a production-ready ASGI server like Gunicorn with Uvicorn workers
   - Configure appropriate worker processes based on available CPU cores
   - Set up monitoring and logging

## Scaling Considerations

- Stateless design allows horizontal scaling
- Consider model size and memory requirements when scaling
- Database or distributed cache may be needed for session state in distributed deployments

## Troubleshooting

Common issues and solutions:

1. **Model Loading Failures**:
   - Verify internet connectivity for model downloads
   - Check available disk space
   - Ensure proper permissions for the artifacts directory

2. **Performance Issues**:
   - Monitor system resources (CPU, memory, disk I/O)
   - Consider batch processing for multiple texts
   - Review model size and hardware requirements

3. **API Errors**:
   - Check request format and parameters
   - Review server logs for detailed error messages
   - Verify axis pack is properly loaded and activated
