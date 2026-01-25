# TensorZero KIE Provider Example

Comprehensive KIE provider example with reasoning, media processing, and streaming responses.

## Overview

KIE provides access to Google Gemini reasoning models optimized for complex problem-solving. This example demonstrates key features of the KIE provider integration with TensorZero:

- **Multiple Model Variants**: Gemini 3 Pro/Flash, Gemini 2.5 Pro/Flash
- **Extended Reasoning**: Configure reasoning effort levels (`low`, `high`)
- **Streaming Support**: Real-time response streaming
- **Media Processing**: Image, video, PDF, and multimodal analysis
- **Thought Blocks**: Access intermediate reasoning steps
- **Tool Calling**: Support for function/tool calling
- **Flexible Configuration**: Mix different KIE models for comparison

## Quick Start

### Prerequisites

1. **KIE API Key**: Set the `KIE_API_KEY` environment variable
   ```bash
   export KIE_API_KEY="your-kie-api-key"
   ```

2. **Start the Gateway**:
   ```bash
   RUST_LOG=debug cargo run -r --bin gateway -- --config-file examples/kie-reasoning/config/tensorzero.toml
   ```

3. **Python 3.10+** or **Node.js 18+** (depending on which example you run)

### Running Tests

Choose your preferred way to test the KIE integration:

**Python (recommended - detailed report)**:
```bash
python3 tests.py
```

**Shell Script**:
```bash
bash tests.sh
```

**Manual curl Commands**:
```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Configuration

KIE provider is configured in `config/tensorzero.toml`. The following models and functions are supported:

### Supported Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `gemini-3-pro` | Medium | Excellent | Complex reasoning |
| `gemini-3-flash` | Fast | Very Good | Balanced use |
| `gemini-2.5-pro` | Medium | Good | Budget optimization |
| `gemini-2.5-flash` | Very Fast | Good | Speed critical |

### Configured Functions

#### 1. Problem Solving
```toml
[functions.solve_problem]
[functions.solve_problem.variants.kie_reasoning]
model = "kie::gemini-3-pro"
max_tokens = 16000
reasoning_effort = "high"

[functions.solve_problem.variants.kie_reasoning_fast]
model = "kie::gemini-3-flash"
max_tokens = 8000
reasoning_effort = "low"
```

#### 2. Code Analysis
```toml
[functions.code_analysis]
[functions.code_analysis.variants.kie_code]
model = "kie::gemini-3-pro"
max_tokens = 12000
reasoning_effort = "high"
```

#### 3. Image Analysis
```toml
[functions.image_analysis]
[functions.image_analysis.variants.kie_vision]
model = "kie::gemini-3-pro"
max_tokens = 4000
reasoning_effort = "high"

[functions.image_analysis.variants.kie_vision_fast]
model = "kie::gemini-3-flash"
max_tokens = 2000
reasoning_effort = "low"
```

#### 4. Video Analysis
```toml
[functions.video_analysis]
[functions.video_analysis.variants.kie_video]
model = "kie::gemini-3-pro"
max_tokens = 6000
reasoning_effort = "high"

[functions.video_analysis.variants.kie_video_2_5]
model = "kie::gemini-2.5-pro"
max_tokens = 6000
reasoning_effort = "high"
```

#### 5. Document Analysis
```toml
[functions.document_analysis]
[functions.document_analysis.variants.kie_document]
model = "kie::gemini-3-pro"
max_tokens = 8000
reasoning_effort = "high"

[functions.document_analysis.variants.kie_document_fast]
model = "kie::gemini-3-flash"
max_tokens = 4000
reasoning_effort = "low"
```

#### 6. Multimodal Reasoning
```toml
[functions.multimodal_reasoning]
[functions.multimodal_reasoning.variants.kie_multimodal]
model = "kie::gemini-3-pro"
max_tokens = 10000
reasoning_effort = "high"

[functions.multimodal_reasoning.variants.kie_multimodal_2_5]
model = "kie::gemini-2.5-pro"
max_tokens = 10000
reasoning_effort = "high"

[functions.multimodal_reasoning.variants.kie_multimodal_fast]
model = "kie::gemini-3-flash"
max_tokens = 6000
reasoning_effort = "low"
```

## Project File Structure

```
examples/kie-reasoning/
├── config/
│   └── tensorzero.toml        # Function and variant configuration
├── tests.py                    # Python test suite
├── tests.sh                    # Shell test script
├── TESTING.md                  # Detailed testing guide
├── README.md                   # This file
├── example.py                  # Python example
├── example.ts                  # TypeScript example
├── QUICKSTART.md               # Quick start guide
└── requirements.txt            # Python dependencies
```

## Usage Examples

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

# Run the example
npx ts-node example.ts

# Or compile and run
npm run build
node dist/example.js
```

## Streaming Responses

KIE provider supports streaming for real-time feedback. See [TESTING.md](TESTING.md) for details.

## Testing

The complete test suite includes:

1. **Basic Reasoning** - High/low reasoning effort
2. **Model Variants** - All 4 Gemini models
3. **Rich Media** - Image, video, PDF analysis
4. **Multimodal** - Multiple media file combination analysis
5. **Function Variants** - TOML configuration function usage
6. **Streaming Response** - Real-time output streaming

Run the tests:
```bash
# Python version (recommended)
python3 tests.py

# Shell version
bash tests.sh
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Supported Media Formats

### Images
- JPEG (`image/jpeg`)
- PNG (`image/png`)
- GIF (`image/gif`)
- WebP (`image/webp`)

### Videos
- MP4 (`video/mp4`)
- WebM (`video/webm`)
- QuickTime (`video/quicktime`)

### Documents
- PDF (`application/pdf`)
- DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`)

## Performance Recommendations

| Task | Recommended Model | Max Tokens | Reasoning Effort |
|------|--------|---------|--------|
| Quick Answer | gemini-3-flash | 2000 | low |
| General Task | gemini-3-pro | 4000 | low |
| Complex Analysis | gemini-3-pro | 8000 | high |
| Image Analysis | kie_vision | 4000 | high |
| Video Analysis | kie_video | 6000 | high |
| Document Analysis | kie_document | 8000 | high |

## Troubleshooting

### Gateway Connection Failed
```
✗ Gateway is not running!
```
**Solution**: Start the gateway
```bash
RUST_LOG=debug cargo run -r --bin gateway -- --config-file examples/kie-reasoning/config/tensorzero.toml
```

### Model Not Found
```
"error": "Model kie::unknown-model not found"
```
**Solution**: Check model names in `config/tensorzero.toml`

### Media Processing Failed
```
"error": "Failed to process media file"
```
**Solution**:
- Verify MIME type is correct
- Ensure URL is accessible
- Check if media format is supported

## Related Documentation

- [TESTING.md](TESTING.md) - Complete testing guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [config/tensorzero.toml](config/tensorzero.toml) - Function configuration
- [../../IMPLEMENTATION_SUMMARY.md](../../IMPLEMENTATION_SUMMARY.md) - Implementation details

## Related Links

- [KIE Official Documentation](https://www.kie.ai/docs)
- [TensorZero Documentation](https://tensorzero.com)
- [Gemini Models](https://ai.google.dev/gemini)
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
