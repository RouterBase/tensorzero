# TensorZero KIE Reasoning Example

This directory contains examples of using TensorZero with the **KIE (Knowledge-Intensive Engine)** provider for advanced reasoning tasks.

## Overview

KIE is an OpenAI-compatible reasoning model optimized for complex problem-solving. This example demonstrates key features of the KIE provider integration with TensorZero:

- **Extended Reasoning**: Configure reasoning effort levels (`low`, `medium`, `high`)
- **Streaming Support**: Real-time response streaming with reasoning content
- **Thought Blocks**: Access intermediate reasoning steps
- **Tool Calling**: Support for function/tool calling capabilities
- **Flexible Configuration**: Mix KIE with other models for comparison

## Prerequisites

1. **KIE API Key**: Set the `KIE_API_KEY` environment variable with your KIE API credentials
2. **TensorZero Gateway**: Running locally or accessible at the configured endpoint
3. **Python 3.10+** or **Node.js 18+** (depending on which example you run)

## Configuration

The KIE provider is configured in `config/tensorzero.toml` with several variants:

### Basic Configuration Example

```toml
[functions.solve_problem.variants.kie_reasoning]
type = "chat_completion"
model = "kie::kie-chat"
max_tokens = 16000
reasoning_effort = "medium"
include_thoughts = true
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier (always `kie::kie-chat`) |
| `max_tokens` | number | Maximum tokens in response (1-32000) |
| `reasoning_effort` | string | `low`, `medium`, or `high` - controls depth of reasoning |
| `include_thoughts` | boolean | Include intermediate reasoning steps in response |
| `temperature` | float | Sampling temperature (0.0 - 2.0) |
| `top_p` | float | Nucleus sampling parameter |
| `stream` | boolean | Enable streaming responses |

## Running the Examples

### Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python example.py
```

### TypeScript/Node.js

```bash
# Install dependencies
npm install

# Run the example (if using ts-node)
npx ts-node example.ts

# Or compile and run
npm run build
node dist/example.js
```

## Examples Included

### Example 1: Problem Solving with Reasoning
Demonstrates streaming responses with medium reasoning effort. Shows how KIE can work through complex problems step-by-step.

### Example 2: Mathematical Reasoning
Uses high reasoning effort to prove mathematical theorems. Demonstrates KIE's capability for rigorous logical reasoning.

### Example 3: Code Analysis
Analyzes code snippets with streaming responses. Useful for real-time feedback applications.

### Example 4: Model Comparison
Compares KIE reasoning with GPT-4 Turbo on the same prompt, allowing you to evaluate the differences.

## Using Streaming Responses

The KIE provider supports streaming for real-time feedback:

### Python
```python
async for chunk in client.chat(
    function_name="solve_problem",
    variant_name="kie_reasoning",
    messages=[{"role": "user", "content": "Your prompt here"}],
    stream=True,
):
    if chunk.text_delta:
        print(chunk.text_delta, end="", flush=True)
    # Access reasoning content via chunk.thought_delta if available
```

### TypeScript
```typescript
const stream = await client.chat({
    functionName: "solve_problem",
    variantName: "kie_reasoning",
    messages: [{ role: "user", content: "Your prompt here" }],
    stream: true,
});

for await (const chunk of stream) {
    if (chunk.textDelta) {
        process.stdout.write(chunk.textDelta);
    }
}
```

## Reasoning Effort Levels

- **`low`**: Quick reasoning with minimal overhead. Good for straightforward tasks.
- **`medium`**: Balanced reasoning. Good for moderately complex tasks (default).
- **`high`**: Extended reasoning with maximum depth. Good for very complex problems.

Higher reasoning effort increases latency and token usage but provides more thorough reasoning.

## System Prompts

The example includes custom system prompts for different tasks:

- `config/math_system_prompt.txt`: Optimized for mathematical reasoning
- `config/code_system_prompt.txt`: Optimized for code analysis and reviews

You can customize these prompts for your specific use cases.

## Advanced Features

### Including Thoughts in Response
Set `include_thoughts = true` to access intermediate reasoning steps. These appear as "thought" content blocks in the response.

### Tool Calling
KIE supports tool/function calling just like OpenAI models. Define tools in your function variants, and KIE will handle tool calling seamlessly.

### Batch Processing
While KIE supports single inference requests, it doesn't support batch operations through this provider. Use individual requests or implement your own batching logic.

## API Response Structure

KIE responses include:

```python
{
    "content": "The main response text",
    "reasoning_content": "Extended reasoning if include_thoughts=true",
    "usage": {
        "completion_tokens": 1234,
        "prompt_tokens": 567,
        "total_tokens": 1801
    },
    "model": "kie-chat",
    "created": "2026-01-20T...",
}
```

## Troubleshooting

### Authentication Issues
- Verify `KIE_API_KEY` environment variable is set correctly
- Check that your KIE API key is valid and has the necessary permissions

### Connection Errors
- Ensure TensorZero gateway is running: `cargo run-e2e` from the project root
- Verify the gateway is accessible at the configured endpoint

### Token Limit Exceeded
- Reduce `max_tokens` parameter
- Use shorter input prompts
- Try lower `reasoning_effort` setting

### Slow Responses
- This is expected with `reasoning_effort = "high"`
- Consider using `"medium"` for better latency/quality tradeoff
- Disable `include_thoughts` if you don't need reasoning steps

## Further Reading

- [TensorZero Documentation](https://www.tensorzero.com/docs)
- [KIE Provider Integration Guide](../../KIE_IMPLEMENTATION_GUIDE.md)
- [Gateway Configuration](../../docs/gateway)
- [OpenAI API Reference](https://platform.openai.com/docs)

## Next Steps

1. Experiment with different `reasoning_effort` levels
2. Create custom system prompts for your domain
3. Integrate KIE into your existing TensorZero applications
4. Monitor performance and costs through the TensorZero UI
5. Set up A/B tests comparing KIE with other models using TensorZero's experimentation features

## Support

For issues or questions:
1. Check the [TensorZero Documentation](https://www.tensorzero.com/docs)
2. Review the [KIE Implementation Guide](../../KIE_IMPLEMENTATION_GUIDE.md)
3. Open an issue on [GitHub](https://github.com/tensorzero/tensorzero)
