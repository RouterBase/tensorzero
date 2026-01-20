## ✅ KIE 示例创建完成

已在 `examples/kie-reasoning` 目录下创建了完整的 KIE 提供程序示例。

### 📁 创建的文件结构

```
examples/kie-reasoning/
├── README.md                      # 详细文档（英文）
├── QUICKSTART.md                  # 快速开始指南（中文）
├── example.py                     # Python 示例代码
├── example.ts                     # TypeScript/Node.js 示例代码
├── package.json                   # Node.js 依赖配置
├── requirements.txt               # Python 依赖配置
├── tsconfig.json                  # TypeScript 编译配置
└── config/
    ├── tensorzero.toml           # KIE 配置文件
    ├── math_system_prompt.txt    # 数学推理系统提示
    └── code_system_prompt.txt    # 代码分析系统提示
```

### 🎯 包含的示例

1. **问题求解** - 演示流式响应和中等推理努力
2. **数学推理** - 使用高推理努力证明数学定理
3. **代码分析** - 实时反馈的代码分析
4. **模型对比** - 与 GPT-4 Turbo 的对比

### 🚀 快速开始

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

### 📋 配置特性

- ✅ 多个 KIE 变体（不同推理努力级别）
- ✅ 流式和非流式推理
- ✅ 思维块/推理内容支持
- ✅ 工具调用基础设施
- ✅ 自定义系统提示
- ✅ 与其他模型的对比

### 🔑 关键配置参数

```toml
model = "kie::kie-chat"           # KIE 模型标识符
max_tokens = 16000                # 最大令牌数
reasoning_effort = "medium"       # "low", "medium", "high"
include_thoughts = true           # 包含推理步骤
stream = true                     # 启用流式响应
```

### 📖 文档

- **README.md** - 完整的英文文档，包括所有功能和参数说明
- **QUICKSTART.md** - 中文快速开始指南，包括故障排除
- 示例代码中包含详细注释

### 参考 DeepSeek 的最佳实践

这个示例跟随 TensorZero 的 DeepSeek 提供程序的模式：
- 使用相同的配置结构
- 演示流式和非流式推理
- 展示如何集成多个变体
- 包含系统提示示例
