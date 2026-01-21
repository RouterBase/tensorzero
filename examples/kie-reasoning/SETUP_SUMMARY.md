# KIE Reasoning Example - Setup Summary

## ✅ KIE Example Setup Complete

A complete KIE (Knowledge-Intensive Engine) provider example has been created in the `examples/kie-reasoning` directory.

### 📁 Directory Structure

```
examples/kie-reasoning/
├── README.md                      # Comprehensive documentation (English)
├── QUICKSTART.md                  # Quick start guide (English)
├── example.py                     # Python example code
├── example.ts                     # TypeScript/Node.js example code
├── package.json                   # Node.js dependencies
├── requirements.txt               # Python dependencies
├── tsconfig.json                  # TypeScript configuration
└── config/
    ├── tensorzero.toml           # KIE configuration
    ├── math_system_prompt.txt    # Math reasoning system prompt
    └── code_system_prompt.txt    # Code analysis system prompt
```

### 🎯 Included Examples

1. **Problem Solving** - Demonstrates streaming responses with medium reasoning effort
2. **Math Reasoning** - Uses high reasoning effort to prove mathematical theorems
3. **Code Analysis** - Real-time streaming feedback for code review
4. **Model Comparison** - Compare KIE with GPT-4 Turbo on same prompts

### 🚀 Quick Start

**Python:**
```bash
export KIE_API_KEY="your-api-key"
pip install -r requirements.txt
python example.py
```

**TypeScript:**
```bash
export KIE_API_KEY="your-api-key"
npm install
npm start
```

### 📋 Configuration Features

- ✅ Multiple KIE variants (different reasoning effort levels)
- ✅ Streaming and non-streaming inference
- ✅ Thought blocks/reasoning content support
- ✅ Tool calling infrastructure
- ✅ Custom system prompts
- ✅ Model comparison setup

### 🔑 Key Configuration Parameters

```toml
model = "kie::gemini-3-pro"           # KIE model identifier
max_tokens = 16000                # Maximum token limit
reasoning_effort = "medium"       # "low", "medium", or "high"
include_thoughts = true           # Include reasoning steps
stream = true                     # Enable streaming responses
```

### 📖 Documentation

- **README.md** - Complete English documentation with all features and parameters
- **QUICKSTART.md** - Quick start guide with troubleshooting
- Detailed code comments in Python and TypeScript examples

### 🎯 Supported Use Cases

1. **Academic Research** - Use `reasoning_effort = "high"` for deep analysis
2. **Real-time Chat** - Use `reasoning_effort = "low"` with streaming
3. **Code Review** - Medium reasoning with custom system prompts
4. **A/B Testing** - Compare different reasoning effort levels

### ✅ Following DeepSeek Best Practices

This example follows TensorZero's DeepSeek provider pattern:
- Uses consistent configuration structure
- Demonstrates both streaming and non-streaming inference
- Shows how to integrate multiple variants
- Includes system prompt templates
- Provides comprehensive documentation and examples
