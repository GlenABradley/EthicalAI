# Corrected Technical Roadmap: Building on Existing EthicalAI Implementation

## Current Implementation Assessment

### ✅ Already Built
1. **Five Ethical Axes** (of 7 planned):
   - Virtue (autonomy-respecting vs eroding)
   - Deontology (rule adherence)
   - Consequentialism (outcome goodness)
   - Intent Bad (manipulative objectives)
   - Intent Good (transparent objectives)

2. **Calibration Data** for 4 dimensions:
   - Autonomy
   - Fairness  
   - Non-aggression
   - Truth

3. **Core Infrastructure**:
   - SentenceTransformer encoder (384-dim embeddings)
   - Axis pack generation and loading
   - Span/frame evaluation system
   - Constitution regularizer for PyTorch
   - API endpoints for axis operations
   - 33+ pre-generated axis packs

### ⚠️ Missing Components for Full Alignment Solution

1. **Two Missing Axes** (to complete the 7):
   - Need to identify and implement the remaining 2 perspectives on "maximize human autonomy from empirical truth"

2. **Training Data Vectorization Pipeline**:
   - Semantic particle extraction
   - Skip-frames and skip-spans implementation
   - Ethical topology mapping

3. **Model Constitution Framework**:
   - Constitutional layer architecture
   - Model surgery tools
   - Training integration

4. **LLM Integration Layer**:
   - Universal adapter system
   - Provider-specific implementations

5. **Runtime Interaction System**:
   - Real-time monitoring
   - Dynamic intervention

## Phase 1: Complete the Seven Axes (Week 1)

### 1.1 Identify Missing Two Axes
Based on existing axes and the core principle, the missing axes likely are:

```python
# configs/axis_packs/causal_coherence.json
{
  "name": "causal_coherence",
  "plain_language_ontology": "Actions aligned with understood consequences vs actions disconnected from causal understanding",
  "plain_language_sought": "Causal disconnect, unintended consequences, butterfly effects ignored, systemic blindness",
  # ... examples focused on causal chains and consequence awareness
}

# configs/axis_packs/information_sovereignty.json  
{
  "name": "information_sovereignty",
  "plain_language_ontology": "Control over one's information environment vs information manipulation/control by others",
  "plain_language_sought": "Information bubbles, echo chambers, algorithmic manipulation, data harvesting",
  # ... examples focused on information autonomy
}
```

### 1.2 Unify Axes Around Core Principle
```python
# src/ethicalai/axes/seven_unified.py
SEVEN_AXES_UNIFIED = {
    "virtue": "Character traits supporting autonomy from truth",
    "deontology": "Rules preserving autonomy through truth",
    "consequentialism": "Outcomes maximizing autonomy via truth",
    "intent": "Objectives aligned with autonomy and truth",
    "causal_coherence": "Actions grounded in truthful causality",
    "information_sovereignty": "Control of truth in one's info space",
    "existential_security": "Long-term autonomy preservation"
}
```

**Deliverables**:
- Create 2 missing axis configuration files
- Generate calibration data for missing axes
- Validate orthogonality of 7-axis system

## Phase 2: Training Data Vectorization (Weeks 2-3)

### 2.1 Semantic Particle System
```python
# src/ethicalai/ml/particles.py
from typing import List, Tuple
import numpy as np

class SemanticParticle:
    """Smallest meaningful unit with ethical vector"""
    def __init__(self, text: str, embedding: np.ndarray):
        self.text = text
        self.embedding = embedding
        self.ethical_scores = self._compute_ethical_scores()
    
    def _compute_ethical_scores(self) -> np.ndarray:
        # Project onto 7 axes using existing axis packs
        return project_to_axes(self.embedding)

class ParticleExtractor:
    def extract_particles(self, text: str) -> List[SemanticParticle]:
        # Token-level extraction
        tokens = self.tokenizer(text)
        # Phrase-level extraction  
        phrases = self.phrase_extractor(text)
        # Sentence-level extraction
        sentences = self.sentence_splitter(text)
        return self._create_particles(tokens + phrases + sentences)
```

### 2.2 Skip-Frames and Skip-Spans
```python
# src/ethicalai/ml/skip_frames.py
class SkipFrame:
    """Non-contiguous semantic relationships"""
    def __init__(self, particles: List[SemanticParticle], skip_pattern: List[int]):
        self.particles = particles
        self.skip_pattern = skip_pattern
        self.topology = self._compute_topology()
    
    def _compute_topology(self) -> np.ndarray:
        # Compute ethical topology across non-contiguous elements
        # This captures long-range dependencies
        pass

class SkipSpan:
    """Spans with gaps for context analysis"""
    def __init__(self, start: int, end: int, skips: List[Tuple[int, int]]):
        self.start = start
        self.end = end
        self.skips = skips  # List of (skip_start, skip_end) tuples
```

### 2.3 Ethical Topology Mapper
```python
# src/ethicalai/ml/topology.py
class EthicalTopologyMapper:
    def __init__(self, axis_packs: Dict[str, AxisPack]):
        self.axis_packs = axis_packs
        self.encoder = get_default_encoder()
    
    def map_dataset(self, dataset: Dataset) -> EthicalTopology:
        """Create 7D ethical landscape of training data"""
        topology = EthicalTopology()
        
        for sample in dataset:
            # Extract all levels
            particles = self.extract_particles(sample)
            frames = self.extract_frames(sample)
            skip_frames = self.extract_skip_frames(sample)
            
            # Project onto 7 axes
            ethical_coords = self.project_to_axes(particles, frames, skip_frames)
            
            # Add to topology
            topology.add_point(ethical_coords, sample)
        
        return topology
```

**Deliverables**:
- Particle extraction pipeline
- Skip-frame/skip-span implementation  
- Dataset topology visualization tools

## Phase 3: Model Constitution Framework (Weeks 4-5)

### 3.1 Constitutional Layer
```python
# src/ethicalai/constitution/layers.py
import torch
import torch.nn as nn

class ConstitutionalLayer(nn.Module):
    """Enforces ethical constraints within model"""
    def __init__(self, model_dim: int, axis_packs: Dict[str, AxisPack]):
        super().__init__()
        self.axis_packs = axis_packs
        self.ethical_projection = nn.Linear(model_dim, 7)
        self.constraint_gates = nn.ModuleList([
            nn.Linear(7, model_dim) for _ in range(7)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Project to ethical space
        ethical_scores = self.ethical_projection(hidden_states)
        
        # Apply constitutional constraints per axis
        constrained = hidden_states
        for i, gate in enumerate(self.constraint_gates):
            axis_constraint = torch.sigmoid(ethical_scores[:, :, i:i+1])
            gated = gate(ethical_scores)
            constrained = constrained + axis_constraint * gated
            
        return constrained
```

### 3.2 Model Surgery
```python
# src/ethicalai/constitution/surgery.py
def inject_ethics_into_model(model: PreTrainedModel, axis_packs: Dict) -> ConstitutionalModel:
    """Inject constitutional layers into existing LLM"""
    
    # Find injection points (after self-attention)
    injection_points = find_attention_outputs(model)
    
    for point in injection_points:
        # Insert constitutional layer
        const_layer = ConstitutionalLayer(
            model.config.hidden_size,
            axis_packs
        )
        inject_after(model, point, const_layer)
    
    # Add ethical loss to training
    model.compute_loss = ethical_loss_wrapper(model.compute_loss)
    
    return ConstitutionalModel(model, axis_packs)
```

**Deliverables**:
- PyTorch constitutional modules
- Model injection utilities
- Training loss modifications

## Phase 4: LLM Integration (Weeks 6-7)

### 4.1 Universal Adapter
```python
# src/ethicalai/integration/universal.py
class UniversalEthicsAdapter:
    """Works with any LLM"""
    
    def __init__(self, axis_packs: Dict[str, AxisPack]):
        self.axis_packs = axis_packs
        self.encoder = get_default_encoder()
        self.topology_mapper = EthicalTopologyMapper(axis_packs)
    
    @classmethod
    def for_transformers(cls, model: PreTrainedModel):
        """HuggingFace Transformers integration"""
        adapter = cls(load_axis_packs("configs/axis_packs"))
        return adapter.inject_into_transformers(model)
    
    @classmethod
    def for_api(cls, provider: str, api_key: str):
        """API-based models (OpenAI, Anthropic)"""
        adapter = cls(load_axis_packs("configs/axis_packs"))
        return adapter.create_api_wrapper(provider, api_key)
    
    def inject_into_transformers(self, model):
        """Direct model modification"""
        return inject_ethics_into_model(model, self.axis_packs)
    
    def create_api_wrapper(self, provider, api_key):
        """Wrapper for API-based models"""
        return EthicalAPIWrapper(provider, api_key, self.axis_packs)
```

### 4.2 API Wrapper
```python
# src/ethicalai/integration/api_wrapper.py
class EthicalAPIWrapper:
    """Wraps API calls with ethical filtering"""
    
    def __init__(self, provider: str, api_key: str, axis_packs: Dict):
        self.client = self._init_client(provider, api_key)
        self.axis_packs = axis_packs
        self.monitor = EthicalMonitor(axis_packs)
    
    async def complete(self, prompt: str, **kwargs):
        # Pre-process: Add ethical framing
        ethical_prompt = self.add_ethical_context(prompt)
        
        # Get completion
        response = await self.client.complete(ethical_prompt, **kwargs)
        
        # Post-process: Evaluate and filter
        filtered = self.filter_response(response)
        
        # Monitor for drift
        self.monitor.track(prompt, filtered)
        
        return filtered
```

**Deliverables**:
- Universal adapter class
- Provider-specific implementations
- API wrapper with filtering

## Phase 5: Runtime Interaction (Weeks 8-9)

### 5.1 Real-time Monitor
```python
# src/ethicalai/runtime/monitor.py
class EthicalMonitor:
    def __init__(self, axis_packs: Dict[str, AxisPack]):
        self.axis_packs = axis_packs
        self.history = ConversationHistory()
        self.trajectory = EthicalTrajectory()
        
    def track(self, user_input: str, model_output: str):
        # Compute ethical scores
        input_scores = self.evaluate(user_input)
        output_scores = self.evaluate(model_output)
        
        # Update trajectory
        self.trajectory.add(input_scores, output_scores)
        
        # Detect concerning patterns
        if self.trajectory.detect_drift():
            return self.intervene()
```

### 5.2 Intervention System
```python
# src/ethicalai/runtime/intervention.py
class InterventionPolicy:
    def __init__(self, strictness: str = "balanced"):
        self.thresholds = STRICTNESS_PRESETS[strictness]
        
    def should_intervene(self, scores: np.ndarray) -> Tuple[bool, str]:
        violations = []
        for i, score in enumerate(scores):
            if score < self.thresholds[i]:
                violations.append((i, score))
        
        if violations:
            severity = self.assess_severity(violations)
            return True, self.generate_intervention(severity, violations)
        return False, ""
```

**Deliverables**:
- Real-time monitoring system
- Intervention policies
- Drift detection

## Phase 6: Evaluation & Deployment (Weeks 10)

### 6.1 Comprehensive Testing
```python
# src/ethicalai/eval/benchmarks.py
class EthicalAlignmentBenchmark:
    def __init__(self):
        self.tests = {
            "truthfulness": TruthfulQA(),
            "autonomy": AutonomyPreservation(),
            "safety": SafetyBenchmark(),
            "consistency": ConsistencyTest(),
            "jailbreak": JailbreakResistance()
        }
    
    def evaluate(self, model) -> Dict[str, float]:
        results = {}
        for name, test in self.tests.items():
            results[name] = test.run(model)
        return results
```

### 6.2 Package & Deploy
```bash
# Installation
pip install ethicalai-alignment

# Quick start
from ethicalai import align

# For local model
model = load_model("llama-2-7b")
ethical_model = align(model)

# For API model
ethical_gpt = align.for_api("openai", api_key)
```

## Implementation Priority

### Week 1: Complete Seven Axes
- [ ] Create 2 missing axis configs
- [ ] Generate calibration data
- [ ] Validate orthogonality

### Weeks 2-3: Training Pipeline
- [ ] Implement particle extraction
- [ ] Build skip-frame system
- [ ] Create topology mapper

### Weeks 4-5: Constitution Framework
- [ ] Build constitutional layers
- [ ] Implement model surgery
- [ ] Integrate with training

### Weeks 6-7: LLM Integration
- [ ] Create universal adapter
- [ ] Build API wrappers
- [ ] Test with major providers

### Weeks 8-9: Runtime System
- [ ] Implement monitoring
- [ ] Build intervention system
- [ ] Add drift detection

### Week 10: Evaluation & Release
- [ ] Run benchmarks
- [ ] Package for distribution
- [ ] Write documentation

## Key Differences from Original Roadmap

1. **Acknowledges existing implementation** - 5 axes, calibration data, infrastructure
2. **Focuses on missing pieces** - 2 axes, training pipeline, model integration
3. **Shorter timeline** - 10 weeks vs 20 weeks
4. **Builds on existing code** - Uses current axis packs, encoder, API structure
5. **Prioritizes integration** - Emphasis on making it work with any LLM

This roadmap respects the significant work already done and focuses on completing the vision of a universal alignment solution.
