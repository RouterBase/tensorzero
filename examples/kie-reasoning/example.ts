#!/usr/bin/env node
/**
 * KIE Reasoning Example (TypeScript/Node.js)
 *
 * This example demonstrates how to use TensorZero with the KIE provider
 * for advanced reasoning tasks.
 *
 * Features demonstrated:
 * - Streaming responses with reasoning content
 * - Non-streaming inference
 * - Switching between different reasoning effort levels
 * - Structured output with tool calling
 */

import { InferenceClient } from "tensorzero";

const client = new InferenceClient();

async function exampleSolveProblem(): Promise<void> {
  const problem = `
    A train leaves City A at 8:00 AM traveling at 60 mph.
    Another train leaves City B (300 miles away) at 9:00 AM traveling at 80 mph toward City A.
    When will they meet? How far from City A?
  `;

  console.log("=".repeat(80));
  console.log("Example 1: Solving a Complex Problem with KIE Reasoning");
  console.log("=".repeat(80));
  console.log(`\nProblem: ${problem.trim()}`);
  console.log("\n--- Using KIE with Medium Reasoning Effort (with streaming) ---\n");

  try {
    // Stream the response
    const stream = await client.chat({
      functionName: "solve_problem",
      variantName: "kie_reasoning",
      messages: [{ role: "user", content: problem }],
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.textDelta) {
        process.stdout.write(chunk.textDelta);
      }
    }
    console.log("\n");
  } catch (error) {
    console.error(`Error during streaming: ${error}`);
  }
}

async function exampleMathReasoning(): Promise<void> {
  const mathProblem = `
    Prove that the sum of the first n odd numbers equals n^2.

    Example:
    - First 1 odd number: 1 = 1^2
    - First 2 odd numbers: 1 + 3 = 4 = 2^2
    - First 3 odd numbers: 1 + 3 + 5 = 9 = 3^2
  `;

  console.log("=".repeat(80));
  console.log("Example 2: Mathematical Reasoning with KIE");
  console.log("=".repeat(80));
  console.log(`\nProblem: ${mathProblem.trim()}`);
  console.log("\n--- Using KIE with High Reasoning Effort ---\n");

  try {
    // Non-streaming for demonstration
    const response = await client.chat({
      functionName: "math_reasoning",
      variantName: "kie_math",
      messages: [{ role: "user", content: mathProblem }],
    });

    console.log(response.content);
    console.log("\n");
  } catch (error) {
    console.error(`Error: ${error}`);
  }
}

async function exampleCodeAnalysis(): Promise<void> {
  const codeSnippet = `
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    result = fibonacci(40)
  `;

  console.log("=".repeat(80));
  console.log("Example 3: Code Analysis with KIE Reasoning");
  console.log("=".repeat(80));
  console.log(`\nCode to analyze:\n${codeSnippet}`);
  console.log("\n--- Using KIE with Streaming for Real-time Feedback ---\n");

  try {
    const stream = await client.chat({
      functionName: "code_analysis",
      variantName: "kie_code_streaming",
      messages: [
        {
          role: "user",
          content: `Please analyze this code and identify any issues:\n${codeSnippet}`,
        },
      ],
      stream: true,
    });

    for await (const chunk of stream) {
      if (chunk.textDelta) {
        process.stdout.write(chunk.textDelta);
      }
    }
    console.log("\n");
  } catch (error) {
    console.error(`Error during streaming: ${error}`);
  }
}

async function exampleCompareModels(): Promise<void> {
  const prompt =
    "Explain how machine learning differs from deep learning in 2-3 sentences.";

  console.log("=".repeat(80));
  console.log("Example 4: Comparing KIE with GPT-4 Turbo");
  console.log("=".repeat(80));
  console.log(`\nPrompt: ${prompt}`);

  console.log("\n--- KIE with Reasoning ---");
  try {
    const response = await client.chat({
      functionName: "solve_problem",
      variantName: "kie_reasoning",
      messages: [{ role: "user", content: prompt }],
    });
    console.log(response.content);
  } catch (error) {
    console.error(`Error: ${error}`);
  }

  console.log("\n--- GPT-4 Turbo (for comparison) ---");
  try {
    const response = await client.chat({
      functionName: "solve_problem",
      variantName: "gpt_4_turbo",
      messages: [{ role: "user", content: prompt }],
    });
    console.log(response.content);
  } catch (error) {
    console.error(`Error: ${error}`);
  }

  console.log("\n");
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("TensorZero KIE Reasoning Examples");
  console.log("=".repeat(80));
  console.log("\nThese examples demonstrate various uses of the KIE provider with TensorZero.");
  console.log("KIE supports:");
  console.log("  - Extended reasoning with configurable reasoning_effort");
  console.log("  - Streaming responses for real-time feedback");
  console.log("  - Including thought/reasoning content in responses");
  console.log("  - Tool calling and function calling capabilities\n");

  // Run examples sequentially
  await exampleSolveProblem();
  await exampleMathReasoning();
  await exampleCodeAnalysis();
  await exampleCompareModels();

  console.log("=".repeat(80));
  console.log("Examples completed!");
  console.log("=".repeat(80));
}

main().catch(console.error);
