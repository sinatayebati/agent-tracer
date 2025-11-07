# Quick Start: Testing Logprobs Implementation

## 🚀 Immediate Testing Guide

### Step 1: Run a Quick Simulation

Run a minimal simulation to test logprobs capture:

```bash
# Using your Gemini/Vertex AI models
tau2 run \
  --domain telecom \
  --agent-llm vertex_ai/gemini-2.5-flash \
  --user-llm vertex_ai/gemini-2.5-flash \
  --num-trials 1 \
  --num-tasks 1 \
  --max-steps 5 \
  --max-concurrency 1
```

```bash
tau2 run \
--domain airline \
--agent-llm vertex_ai/gemini-2.5-flash \
--user-llm vertex_ai/gemini-2.5-flash \
--num-trials 1 \
--num-tasks 1 \
--max-concurrency 1
```

This will:
- Run 1 task from the telecom domain
- Limit to 1 trial and 5 steps (fast completion)
- Use your Gemini models for both agent and user
- Save results to `data/simulations/`

### Step 2: Verify Logprobs Were Captured

Run the verification script on the output:

```bash
# The simulation file will be saved with a timestamp
# Find the most recent file:
ls -lt data/simulations/*.json | head -1

# Or run verification on the latest file:
python verify_logprobs.py $(ls -t data/simulations/*.json | head -1)
```

### Expected Output

You should see something like:

```
======================================================================
LOGPROBS VERIFICATION RESULTS
======================================================================

File: data/simulations/2025-11-07T...telecom_llm_agent_gemini...json

Simulations analyzed: 1
Total messages (agent + user): 10

--- Logprobs Coverage ---
Messages WITH logprobs: 10
Messages WITHOUT logprobs: 0
Coverage: 100.0%

--- By Actor ---
Agent messages with logprobs: 5
User messages with logprobs: 5

======================================================================
✅ SUCCESS: All messages have logprobs!
======================================================================
```

## 🔍 Manual Inspection (Optional)

To manually inspect the logprobs in the JSON:

```bash
# View the structure of one message with logprobs
cat data/simulations/[YOUR_FILE].json | \
  jq '.simulations[0].messages[] | 
      select(.role=="assistant") | 
      {role, content: .content[0:50], has_logprobs: (.logprobs != null)} | 
      select(.has_logprobs)' | \
  head -20
```

To see the actual logprobs data:

```bash
# View detailed logprobs from first assistant message
cat data/simulations/[YOUR_FILE].json | \
  jq '.simulations[0].messages[] | 
      select(.role=="assistant") | 
      .logprobs' | \
  head -1
```

## ✅ Success Criteria

Your implementation is working correctly if:

1. ✅ The simulation completes without errors
2. ✅ The verification script shows 100% coverage
3. ✅ Both agent and user messages have logprobs
4. ✅ The `logprobs` field in the JSON is not `null`

## ⚠️ Troubleshooting

### Issue: "No logprobs found in any messages"

**Solution**: Check if your API/model supports logprobs:

```bash
# Test with a known-compatible model first
tau2 run \
  --domain telecom \
  --num-tasks 1 \
  --num-trials 1 \
  --max-steps 5 \
  --llm-agent gpt-4 \
  --llm-user gpt-4
```

### Issue: Partial logprobs coverage

**Solution**: Some models may not return logprobs for certain types of responses (e.g., tool calls). Check:

```bash
# See which messages are missing logprobs
python verify_logprobs.py [YOUR_FILE].json
# Then inspect those specific messages in the JSON
```

### Issue: Simulation fails

**Solution**: Check logs for API errors:

```bash
# Run with debug logging
tau2 run \
  --domain telecom \
  --num-tasks 1 \
  --log-level DEBUG \
  --llm-agent vertex_ai/gemini-2.0-flash-exp \
  --llm-user vertex_ai/gemini-2.0-flash-exp
```

## 📊 Next: Analyzing Logprobs

Once verified, you can analyze the logprobs:

```python
from tau2.data_model.simulation import Results
import numpy as np

# Load results
results = Results.load("data/simulations/[YOUR_FILE].json")

# Calculate average log probability per message
for sim in results.simulations:
    print(f"\nSimulation: {sim.id}")
    for i, msg in enumerate(sim.messages):
        if msg.role in ['assistant', 'user'] and msg.logprobs:
            # Extract token logprobs (format depends on model)
            if 'content' in msg.logprobs:
                logprobs_values = [
                    token['logprob'] 
                    for token in msg.logprobs['content'] 
                    if 'logprob' in token
                ]
                if logprobs_values:
                    avg_logprob = np.mean(logprobs_values)
                    uncertainty = -avg_logprob  # Higher = more uncertain
                    print(f"  Msg {i} ({msg.role}): "
                          f"avg_logprob={avg_logprob:.3f}, "
                          f"uncertainty={uncertainty:.3f}")
```

## 🎯 Production Use

For production runs with full task sets:

```bash
# Remove the limitations
tau2 run \
  --domain telecom \
  --num-trials 3 \
  --llm-agent vertex_ai/gemini-2.0-flash-exp \
  --llm-user vertex_ai/gemini-2.0-flash-exp
```

All simulations will now automatically capture logprobs! 🎉

## 📝 Notes

- **Performance**: Logprobs add minimal overhead (~2-5% slower)
- **Storage**: JSON files will be ~20-30% larger with logprobs
- **Compatibility**: Works with any LiteLLM-compatible model that supports logprobs
- **Automatic**: No configuration needed - logprobs are requested by default

