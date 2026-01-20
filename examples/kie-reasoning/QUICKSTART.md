# KIE Reasoning Example - Quick Start Guide

Get started with the KIE (Knowledge-Intensive Engine) provider in TensorZero in just a few minutes.

## What is KIE?

KIE is an OpenAI-compatible reasoning model optimized for:
- **Complex Problem Solving** - Deep reasoning for challenging tasks
- **Extended Thinking** - Configurable reasoning effort levels
- **Real-time Streaming** - Stream responses as they're generated
- **Tool Integration** - Seamless function/tool calling support

## Prerequisites

### 1. Get Your KIE API Key

You'll need a KIE API key to run these examples. If you don't have one, visit [KIE's website](https://www.kie.com) to sign up.

### 2. Set Up Environment Variable

```bash
export KIE_API_KEY="your-actual-api-key-here"
```

Or add it to your `.env` file:
```
KIE_API_KEY=your-actual-api-key-here
```

### 3. Start TensorZero Gateway

The examples assume TensorZero gateway is running locally. Start it with:

```bash
# From the project root
cargo run-e2e
```

This will start the gateway on `http://localhost:3000` by default.

## Quick Run - Python

### Install Dependencies

```bash
# Navigate to the example directory
cd examples/kie-reasoning

# Install Python dependencies
pip install -r requirements.txt
```

### Run the Example

```bash
python example.py
```

**Expected Output:**
```
================================================================================
TensorZero KIE Reasoning Examples
================================================================================

These examples demonstrate various uses of the KIE provider with TensorZero.
KIE supports:
  - Extended reasoning with configurable reasoning_effort
  - Streaming responses for real-time feedback
  - Including thought/reasoning content in responses
  - Tool calling and function calling capabilities

================================================================================
Example 1: Solving a Complex Problem with KIE Reasoning
================================================================================

Problem: ...

--- Using KIE with Medium Reasoning Effort (with streaming) ---

[Streaming response appears here...]
```

## Quick Run - TypeScript/Node.js

### Install Dependencies

```bash
# Navigate to the example directory
cd examples/kie-reasoning

# Install Node.js dependencies
npm install
```

### Run the Example

```bash
npm start
```

Or compile and run:
```bash
npm run build
npm run
```

## Configuration Files

The example uses `config/tensorzero.toml` which defines:

### Problem Solving Functions

```toml
[functions.solve_problem.variants.kie_reasoning]
type = "chat_completion"
model = "kie::kie-chat"
max_tokens = 16000
reasoning_effort = "medium"
include_thoughts = true
```

### Math Reasoning

```toml
[functions.math_reasoning.variants.kie_math]
type = "chat_completion"
model = "kie::kie-chat"
max_tokens = 8000
reasoning_effort = "high"
include_thoughts = true
system_instructions = "config/math_system_prompt.txt"
```

### Code Analysis

```toml
[functions.code_analysis.variants.kie_code_streaming]
type = "chat_completion"
model = "kie::kie-chat"
max_tokens = 12000
reasoning_effort = "medium"
stream = true
```

## Understanding Reasoning Effort Levels

KIE supports three reasoning effort levels:

| Level | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **low** | ⚡ Very Fast | Good | Simple questions, quick answers |
| **medium** | 🚀 Moderate | Better | Most tasks, balanced tradeoff |
| **high** | 🐌 Slow | Best | Complex problems, thorough analysis |

### Example: Changing Reasoning Effort

In `config/tensorzero.toml`:

```toml
# Fast reasoning
[functions.solve_problem.variants.kie_fast]
reasoning_effort = "low"
max_tokens = 4000

# Balanced reasoning
[functions.solve_problem.variants.kie_reasoning]
reasoning_effort = "medium"
max_tokens = 16000

# Deep reasoning
[functions.solve_problem.variants.kie_extended]
reasoning_effort = "high"
max_tokens = 32000
```

## Streaming Responses

Stream responses to get real-time feedback:

### Python

```python
async for chunk in client.chat(
    function_name="solve_problem",
    variant_name="kie_reasoning",
    messages=[{"role": "user", "content": "Your question here"}],
    stream=True,
):
    if chunk.text_delta:
        print(chunk.text_delta, end="", flush=True)
```

### TypeScript

```typescript
const stream = await client.chat({
    functionName: "solve_problem",
    variantName: "kie_reasoning",
    messages: [{ role: "user", content: "Your question here" }],
    stream: true,
});

for await (const chunk of stream) {
    if (chunk.textDelta) {
        process.stdout.write(chunk.textDelta);
    }
}
```

## Accessing Reasoning Content

When `include_thoughts = true`, you can access reasoning steps:

### Python

```python
response = await client.chat(
    function_name="math_reasoning",
    variant_name="kie_math",
    messages=[{"role": "user", "content": "Prove that..."}],
)

# Access reasoning content (implementation depends on client version)
if hasattr(response, 'reasoning_content'):
    print("Reasoning:", response.reasoning_content)
```

### TypeScript

```typescript
const response = await client.chat({
    functionName: "math_reasoning",
    variantName: "kie_math",
    messages: [{ role: "user", content: "Prove that..." }],
});

// Access reasoning content
if (response.reasoningContent) {
    console.log("Reasoning:", response.reasoningContent);
}
```

## Customizing System Prompts

Edit system prompt files to customize behavior:

### Math Reasoning (`config/math_system_prompt.txt`)

```
You are an expert mathematics tutor. Your task is to:
1. Understand the problem clearly
2. Provide step-by-step reasoning
3. Show all work and explain each step
...
```

### Code Analysis (`config/code_system_prompt.txt`)

```
You are an expert code reviewer and software engineer. Your task is to:
1. Analyze the provided code carefully
2. Provide detailed insights about its functionality
3. Identify potential bugs and security concerns
...
```

## Comparing Models

Use the example to compare KIE with other models:

```toml
[functions.solve_problem.variants.kie_reasoning]
type = "chat_completion"
model = "kie::kie-chat"
max_tokens = 16000

[functions.solve_problem.variants.gpt_4_turbo]
type = "chat_completion"
model = "openai::gpt-4-turbo"
max_tokens = 16000
```

Then run the same prompt against both variants to compare results.

## Troubleshooting

### Error: "API Key Missing"

**Problem:** `ApiKeyMissing` error when running examples

**Solution:**
```bash
# Make sure KIE_API_KEY is set
echo $KIE_API_KEY

# If empty, set it
export KIE_API_KEY="your-api-key"
```

### Error: "Connection Refused"

**Problem:** `Connection refused` when connecting to gateway

**Solution:**
```bash
# Make sure gateway is running
cargo run-e2e

# Or check if it's already running
curl http://localhost:3000/health
```

### Error: "Model Not Found"

**Problem:** `Model not found` error

**Solution:**
- Verify `model = "kie::kie-chat"` in tensorzero.toml
- Check that KIE API endpoint is correct
- Ensure your API key has access to this model

### Slow Responses

**Problem:** Responses are very slow

**Solution:**
- Try lower `reasoning_effort`: `"low"` or `"medium"`
- This is expected with `reasoning_effort = "high"`
- Reduce `max_tokens` if not needed

### Streaming Not Working

**Problem:** No streaming output

**Solution:**
- Ensure `stream = true` in configuration OR `stream=True` in code
- Check that the model supports streaming (KIE does)
- Use `await client.chat(..., stream=True)` for Python
- Use `stream: true` parameter for TypeScript

## Next Steps

1. **Read Full Documentation**: See [README.md](README.md) for comprehensive docs
2. **Experiment with Parameters**:
   - Try different `reasoning_effort` levels
   - Adjust `max_tokens` values
   - Modify `temperature` parameter
3. **Customize System Prompts**: Edit `config/*.txt` files for your use case
4. **Run A/B Tests**: Compare KIE variants using TensorZero's experimentation features
5. **Monitor Performance**: Use TensorZero UI to track latency and costs

## Example Use Cases

### Academic Research
Use `reasoning_effort = "high"` with math/science reasoning:
```toml
[functions.research_assistant.variants.deep_reasoning]
reasoning_effort = "high"
max_tokens = 32000
include_thoughts = true
```

### Real-time Chat Application
Use `reasoning_effort = "low"` with streaming for responsiveness:
```toml
[functions.chat.variants.fast_response]
reasoning_effort = "low"
stream = true
max_tokens = 4000
```

### Code Review Assistant
Use medium reasoning with custom system prompt:
```toml
[functions.code_review.variants.assistant]
reasoning_effort = "medium"
system_instructions = "config/code_system_prompt.txt"
include_thoughts = true
```

## Support and Resources

- **Full Documentation**: [README.md](README.md)
- **Configuration Guide**: [tensorzero.toml](config/tensorzero.toml)
- **TensorZero Docs**: https://www.tensorzero.com/docs
- **GitHub Issues**: https://github.com/tensorzero/tensorzero/issues

## Performance Tips

1. **Batch Requests**: Send multiple requests together when possible
2. **Cache Results**: Store responses for similar queries
3. **Monitor Costs**: Track API usage in TensorZero UI
4. **Optimize Tokens**: Start with lower `max_tokens`, increase if needed
5. **Test Reasoning Effort**: Find the sweet spot between quality and latency

---

**Happy reasoning! 🚀**

For questions or issues, check the full [README.md](README.md) or visit the [TensorZero documentation](https://www.tensorzero.com/docs).
3. 在 [GitHub](https://github.com/tensorzero/tensorzero) 上开启 Issue

---

**祝你使用愉快！** 🎉
