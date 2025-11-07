# Quick Start: Step 2 - Uncertainty Calculation

## 🚀 Get Started in 3 Steps

### Step 1: Run Unit Tests

Verify the implementation works correctly:

```bash
python test_step2.py
```

**Expected Output:**
```
🧪 TESTING STEP 2: UNCERTAINTY CALCULATION
============================================================
TEST 1: Normalized Entropy Calculation
...
============================================================
TEST SUMMARY
============================================================
Normalized Entropy: ✅ PASS
Edge Cases: ✅ PASS
Response Statistics: ✅ PASS
Token-Level Uncertainties: ✅ PASS
Real-World Data: ✅ PASS

============================================================
🎉 ALL TESTS PASSED!
Step 2 implementation is working correctly.
============================================================
```

### Step 2: Process Your Simulation Data

Run the processing script on your existing simulation:

```bash
python process_trajectories.py data/simulations/2025-11-06T23:54:05.344873_airline_llm_agent_gemini-2.5-flash_user_simulator_gemini-2.5-flash.json
```

**You'll see:**
```
================================================================================
TRAJECTORY UNCERTAINTY ANALYSIS SUMMARY
================================================================================

Source: 2025-11-06T23:54:05.344873
Total Simulations: 1

--- Overall Statistics ---
Agent Reasoning Uncertainty (U_i,agent):
  Mean: 0.1234
  Std:  0.0456
  Min:  0.0234
  Max:  0.4567

User Confusion (U_i,user):
  Mean: 0.2345
  Std:  0.0678
  Min:  0.0567
  Max:  0.5678
...
```

### Step 3: View Detailed Turn-by-Turn Analysis

See exactly what's happening at each turn:

```bash
python process_trajectories.py data/simulations/2025-11-06T23:54:05.344873_airline_llm_agent_gemini-2.5-flash_user_simulator_gemini-2.5-flash.json --detailed
```

**You'll see:**
```
================================================================================
DETAILED TRAJECTORY: Simulation 1
================================================================================
...
--- Turn-by-Turn Uncertainty (U_i) ---

Turn   Actor      U_i Score    Content Preview
--------------------------------------------------------------------------------
1      user       0.2341       Hi there! I'm calling because I need to cancel...
2      agent      0.0891       I can help you with that. What is your user ID...
3      user       0.1567       My user ID is U12345...
4      agent      0.1023       I understand. Let me process the cancellation...
...
```

## 📊 Export Results for Analysis

Save processed results to JSON:

```bash
python process_trajectories.py \
  data/simulations/2025-11-06T23:54:05.344873_airline_llm_agent_gemini-2.5-flash_user_simulator_gemini-2.5-flash.json \
  --output results/uncertainty_analysis.json \
  --verbose
```

This creates a JSON file with:
- Turn-by-turn U_i scores
- Per-simulation summaries
- Detailed statistics (with `--verbose`)

## 🔍 Using in Python

### Quick Analysis Script

```python
import json
from saup_metrics import calculate_normalized_entropy

# Load your simulation
with open('data/simulations/your_file.json', 'r') as f:
    data = json.load(f)

# Analyze first simulation
sim = data['simulations'][0]
print(f"Task: {sim['task_id']}\n")

for msg in sim['messages']:
    if msg['role'] in ['assistant', 'user']:
        actor = 'agent' if msg['role'] == 'assistant' else 'user'
        ui = calculate_normalized_entropy(msg.get('logprobs'))
        content = msg.get('content', '')[:50]
        print(f"{actor:8} U_i={ui:.4f} | {content}...")
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from process_trajectories import load_simulation_results, process_all_simulations

# Process all simulations in directory
sim_dir = Path('data/simulations')
results_dir = Path('results/uncertainty')
results_dir.mkdir(exist_ok=True)

for sim_file in sim_dir.glob('*.json'):
    print(f"Processing {sim_file.name}...")
    
    data = load_simulation_results(str(sim_file))
    results = process_all_simulations(data, verbose=True)
    
    # Save results
    output_file = results_dir / f"uncertainty_{sim_file.name}"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✅ Saved to {output_file}")
```

## 📈 Understanding the Output

### U_i Score Interpretation

| U_i Value | Interpretation | Example |
|-----------|---------------|---------|
| 0.01 - 0.10 | Very confident | Factual responses, IDs, confirmations |
| 0.10 - 0.30 | Normal confidence | Standard conversation |
| 0.30 - 0.60 | Moderate uncertainty | Complex reasoning, decisions |
| 0.60 - 1.00 | High uncertainty | Confusion, unclear instructions |
| > 1.00 | Very high uncertainty | Model struggling significantly |

### Key Metrics

**Per-Turn:**
- `ui_score`: The normalized entropy for this turn
- `actor`: 'agent' or 'user' (who generated this response)

**Per-Simulation:**
- `mean_uncertainty_agent`: Average agent uncertainty
- `mean_uncertainty_user`: Average user uncertainty
- `max_uncertainty_overall`: Peak uncertainty in trajectory

## 🎯 Common Use Cases

### 1. Identify High-Uncertainty Turns

```bash
# Process with detailed output
python process_trajectories.py simulation.json --detailed

# Look for turns with U_i > 0.5 (high uncertainty)
```

### 2. Compare Agent vs User Uncertainty

```python
from process_trajectories import process_all_simulations

results = process_all_simulations(data)

for sim in results['results']:
    agent_unc = sim['summary']['mean_uncertainty_agent']
    user_unc = sim['summary']['mean_uncertainty_user']
    
    print(f"Task {sim['task_id']}:")
    print(f"  Agent: {agent_unc:.4f}")
    print(f"  User:  {user_unc:.4f}")
    print(f"  Ratio: {user_unc/agent_unc:.2f}x")
```

### 3. Track Uncertainty Over Time

```python
# Plot uncertainty trajectory
import matplotlib.pyplot as plt

sim = results['results'][0]
turns = [s['turn'] for s in sim['uncertainty_scores']]
uncertainties = [s['ui_score'] for s in sim['uncertainty_scores']]
actors = [s['actor'] for s in sim['uncertainty_scores']]

# Separate agent and user
agent_turns = [t for t, a in zip(turns, actors) if a == 'agent']
agent_unc = [u for u, a in zip(uncertainties, actors) if a == 'agent']
user_turns = [t for t, a in zip(turns, actors) if a == 'user']
user_unc = [u for u, a in zip(uncertainties, actors) if a == 'user']

plt.plot(agent_turns, agent_unc, 'o-', label='Agent')
plt.plot(user_turns, user_unc, 's-', label='User')
plt.xlabel('Turn')
plt.ylabel('Uncertainty (U_i)')
plt.legend()
plt.title(f"Uncertainty Trajectory: {sim['task_id']}")
plt.show()
```

## ⚠️ Troubleshooting

### Issue: "No logprobs found"

**Solution:** Make sure you ran Step 1 and generated new simulations:
```bash
# Generate new simulation with logprobs
tau2 run --domain airline --num-tasks 1 \
  --llm-agent vertex_ai/gemini-2.0-flash-exp \
  --llm-user vertex_ai/gemini-2.0-flash-exp
```

### Issue: U_i values seem wrong

**Solution:** Check if logprobs are actually present:
```bash
# Verify logprobs exist
cat simulation.json | jq '.simulations[0].messages[1].logprobs.content[0]'
```

### Issue: Script crashes

**Solution:** Check file format:
```bash
# Validate JSON
python -m json.tool simulation.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

## 📚 Next Steps

Once you've processed your simulations:

1. **Analyze patterns**: Which tasks have high uncertainty?
2. **Compare models**: Do different models show different uncertainty patterns?
3. **Identify failures**: Do high U_i scores correlate with task failures?
4. **Move to Step 3**: Implement temporal uncertainty propagation

## 🔗 Quick Reference

**Files Created:**
- `saup_metrics.py` - Core metrics functions
- `process_trajectories.py` - CLI processing tool
- `test_step2.py` - Unit tests
- `STEP2_UNCERTAINTY_CALCULATION.md` - Full documentation

**Key Commands:**
```bash
# Test implementation
python test_step2.py

# Process single file
python process_trajectories.py simulation.json

# Detailed view
python process_trajectories.py simulation.json --detailed

# Export to JSON
python process_trajectories.py simulation.json -o results.json -v
```

**Import in Python:**
```python
from saup_metrics import calculate_normalized_entropy
from process_trajectories import process_all_simulations
```

Ready to process your trajectories! 🚀

