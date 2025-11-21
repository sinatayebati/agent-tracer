# Quick Reference: Uncertainty Analysis

## 🚀 Common Commands

### Run Simulations

* Basic simulation w/ log probs only.

```bash
# Using your Gemini/Vertex AI models
tau2 run \
  --domain airline \
  --num-tasks 2 \
  --num-trials 1 \
  --max-steps 5 \
  --agent-llm vertex_ai/gemini-2.5-flash \
  --user-llm vertex_ai/gemini-2.5-flash \
```

* Advance simulation w/ uncertainty calculation

```bash
tau2 run \
  --domain airline \
  --num-tasks 2 \
  --num-trials 1 \
  --agent-llm vertex_ai/gemini-2.5-flash \
  --user-llm vertex_ai/gemini-2.5-flash \
  --calculate-uncertainty
```

### Run separate uncertainty tests
```bash
# Simple test runner (no dependencies needed)
python tests/run_uncertainty_tests.py

# With pytest (if installed)
pytest tests/test_uncertainty.py -v
```

### Analyze Simulations
```bash
# Basic analysis (auto-saves to data/uncertainty/)
python -m src.tau2.scripts.analyze_uncertainty data/simulations/your_file.json

# Detailed turn-by-turn view (auto-saves)
python -m src.tau2.scripts.analyze_uncertainty data/simulations/your_file.json --detailed

# Include verbose statistics in output
python -m src.tau2.scripts.analyze_uncertainty data/simulations/your_file.json --verbose

# Custom output location
python -m src.tau2.scripts.analyze_uncertainty data/simulations/your_file.json \
  --output results/custom_analysis.json

# Don't save, just display results
python -m src.tau2.scripts.analyze_uncertainty data/simulations/your_file.json --no-save
```

**Note**: By default, results are automatically saved to `data/uncertainty/` with the same filename as your simulation file for easy cross-referencing.

### Batch Process All Simulations
```bash
# Analyze all simulations at once (skips already analyzed files)
./scripts/batch_analyze_uncertainty.sh

# With verbose statistics
./scripts/batch_analyze_uncertainty.sh --verbose

# With detailed view for each
./scripts/batch_analyze_uncertainty.sh --detailed
```

## 📝 Code Examples

### Calculate Uncertainty for a Message

```python
from tau2.metrics import calculate_normalized_entropy

# Your message with logprobs
message = {
    "role": "assistant",
    "content": "I can help you with that.",
    "logprobs": {
        "content": [
            {"token": "I", "logprob": -0.074},
            {"token": " can", "logprob": -0.0008},
            # ... more tokens
        ]
    }
}

# Calculate uncertainty
ui = calculate_normalized_entropy(message['logprobs'])
print(f"Uncertainty (U_i): {ui:.4f}")
```

### Get Detailed Statistics

```python
from tau2.metrics import get_uncertainty_stats

stats = get_uncertainty_stats(message['logprobs'])
print(f"Normalized Entropy: {stats.normalized_entropy:.4f}")
print(f"Token Count: {stats.token_count}")
print(f"Mean Probability: {stats.mean_probability:.4f}")
print(f"Max Uncertainty: {stats.max_uncertainty:.4f}")
```

### Access Uncertainty from Real-Time Simulation

If you ran simulation with `--calculate-uncertainty`, uncertainty stats are already embedded:

```python
from tau2.data_model.simulation import Results

# Load simulation (run with --calculate-uncertainty)
results = Results.load("data/simulations/your_file.json")

# Access embedded uncertainty statistics
for sim in results.simulations:
    for msg in sim.messages:
        if msg.uncertainty is not None:
            u = msg.uncertainty
            print(f"{msg.role}:")
            print(f"  U_i (normalized_entropy): {u['normalized_entropy']:.4f}")
            print(f"  Tokens: {u['token_count']}")
            print(f"  Mean probability: {u['mean_probability']:.4f}")
            print(f"  Std uncertainty: {u['std_uncertainty']:.4f}")
```

### Analyze a Simulation File (Post-Processing)

For simulations run WITHOUT `--calculate-uncertainty`:

```python
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.scripts.analyze_uncertainty import analyze_results

# Load simulation
sim_path = Path("data/simulations/your_file.json")
results = Results.load(sim_path)

# Analyze
analysis = analyze_results(results, verbose=True)

# Access results
for sim in analysis.results:
    print(f"Task: {sim.task_id}")
    print(f"  Agent uncertainty: {sim.summary['mean_uncertainty_agent']:.4f}")
    print(f"  User uncertainty: {sim.summary['mean_uncertainty_user']:.4f}")
```

### Process Multiple Files

```python
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.scripts.analyze_uncertainty import analyze_results
import json

sim_dir = Path("data/simulations")
results_dir = Path("results/uncertainty")
results_dir.mkdir(exist_ok=True)

for sim_file in sim_dir.glob("*.json"):
    print(f"Processing {sim_file.name}...")
    
    # Load and analyze
    results = Results.load(sim_file)
    analysis = analyze_results(results, verbose=True)
    
    # Save
    output_file = results_dir / f"uncertainty_{sim_file.name}"
    with open(output_file, 'w') as f:
        json.dump(analysis.model_dump(), f, indent=2)
    
    print(f"  ✅ Saved to {output_file}")
```

## 📁 File Locations

| What | Where |
|------|-------|
| Core metrics | `src/tau2/metrics/uncertainty.py` |
| CLI tool | `src/tau2/scripts/analyze_uncertainty.py` |
| Tests (pytest) | `tests/test_uncertainty.py` |
| Tests (simple) | `tests/run_uncertainty_tests.py` |
| Documentation | `STEP2_UNCERTAINTY_CALCULATION.md` |

## 🔧 Import Patterns

```python
# Import specific functions
from tau2.metrics import (
    calculate_normalized_entropy,
    get_uncertainty_stats,
    calculate_token_uncertainties,
)

# Or import from specific module
from tau2.metrics.uncertainty import calculate_normalized_entropy

# Import data models
from tau2.metrics import TokenUncertainty, UncertaintyStats
```

## 📊 Understanding U_i Values

| Range | Interpretation | Typical Examples |
|-------|----------------|------------------|
| 0.00 - 0.10 | Very confident | IDs, confirmations, facts |
| 0.10 - 0.30 | Normal | Standard conversation |
| 0.30 - 0.60 | Moderate uncertainty | Complex reasoning |
| 0.60 - 1.00 | High uncertainty | Confusion, ambiguity |
| > 1.00 | Very high uncertainty | Model struggling |

## 🐛 Troubleshooting

### Import Error: "No module named 'tau2'"

**Solution**: Make sure you're in the project directory and the package is installed:
```bash
cd /path/to/agent-uncertainty
pip install -e .
# or
pdm install
```

### Tests Fail

**Solution**: Use the simple test runner that doesn't require pytest:
```bash
python tests/run_uncertainty_tests.py
```

### CLI Script Doesn't Run

**Solution**: Run with PYTHONPATH:
```bash
PYTHONPATH=src python -m tau2.scripts.analyze_uncertainty simulation.json
```

## 💡 Tips

1. **Batch Processing**: Use a loop to process all simulations in a directory
2. **Filtering**: Focus on high-uncertainty turns (U_i > 0.5) to find problems
3. **Comparison**: Compare agent vs user uncertainty ratios
4. **Visualization**: Export to JSON and visualize with matplotlib/seaborn
5. **Thresholds**: Set domain-specific thresholds for intervention

## 📈 Common Analysis Patterns

### Find High-Uncertainty Moments
```python
for sim in analysis.results:
    for turn in sim.uncertainty_scores:
        if turn.ui_score > 0.5:
            print(f"⚠️  High uncertainty at turn {turn.turn}")
            print(f"   Actor: {turn.actor}, U_i: {turn.ui_score:.4f}")
```

### Compare Models
```python
# After analyzing simulations with different models
gpt4_analysis = analyze_results(gpt4_results)
gemini_analysis = analyze_results(gemini_results)

print(f"GPT-4 agent uncertainty: {gpt4_analysis.results[0].summary['mean_uncertainty_agent']:.4f}")
print(f"Gemini agent uncertainty: {gemini_analysis.results[0].summary['mean_uncertainty_agent']:.4f}")
```

### Track Uncertainty Over Trajectory
```python
sim = analysis.results[0]
for turn in sim.uncertainty_scores:
    print(f"Turn {turn.turn:2d} ({turn.actor:5s}): U_i = {turn.ui_score:.4f}")
```

---

**For detailed documentation, see**: `STEP2_UNCERTAINTY_CALCULATION.md`  
**For implementation details, see**: `REFACTORING_SUMMARY.md`

