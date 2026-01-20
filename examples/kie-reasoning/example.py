#!/usr/bin/env python3
"""
KIE Reasoning Example

This example demonstrates how to use TensorZero with the KIE (Knowledge-Intensive Engine)
provider for advanced reasoning tasks. KIE is optimized for complex problem-solving with
extended reasoning capabilities.

Features demonstrated:
- Streaming responses with reasoning content
- Non-streaming inference with metadata
- Switching between different reasoning effort levels
- Comparing KIE reasoning with standard models
"""

import asyncio
from tensorzero import Inference, InferenceClient


async def example_solve_problem():
    """Example: Solve a complex problem using KIE reasoning."""
    client = InferenceClient()

    problem = """
    A train leaves City A at 8:00 AM traveling at 60 mph. 
    Another train leaves City B (300 miles away) at 9:00 AM traveling at 80 mph toward City A.
    When will they meet? How far from City A?
    """

    print("=" * 80)
    print("Example 1: Solving a Complex Problem with KIE Reasoning")
    print("=" * 80)
    print(f"\nProblem: {problem.strip()}")
    print("\n--- Using KIE with Medium Reasoning Effort (with streaming) ---\n")

    # Stream the response
    try:
        async for chunk in client.chat(
            function_name="solve_problem",
            variant_name="kie_reasoning",
            messages=[{"role": "user", "content": problem}],
            stream=True,
        ):
            if chunk.text_delta:
                print(chunk.text_delta, end="", flush=True)
            # In real applications, you could also access chunk.thought_delta for reasoning content
        print("\n")
    except Exception as e:
        print(f"Error during streaming: {e}")


async def example_math_reasoning():
    """Example: Solve a math problem with step-by-step reasoning."""
    client = InferenceClient()

    math_problem = """
    Prove that the sum of the first n odd numbers equals n^2.
    
    Example: 
    - First 1 odd number: 1 = 1^2
    - First 2 odd numbers: 1 + 3 = 4 = 2^2
    - First 3 odd numbers: 1 + 3 + 5 = 9 = 3^2
    """

    print("=" * 80)
    print("Example 2: Mathematical Reasoning with KIE")
    print("=" * 80)
    print(f"\nProblem: {math_problem.strip()}")
    print("\n--- Using KIE with High Reasoning Effort ---\n")

    try:
        # Non-streaming for demonstration
        response = await client.chat(
            function_name="math_reasoning",
            variant_name="kie_math",
            messages=[{"role": "user", "content": math_problem}],
        )
        print(response.content)
        print("\n")
    except Exception as e:
        print(f"Error: {e}")


async def example_code_analysis():
    """Example: Analyze code with KIE reasoning."""
    client = InferenceClient()

    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    result = fibonacci(40)
    """

    print("=" * 80)
    print("Example 3: Code Analysis with KIE Reasoning")
    print("=" * 80)
    print(f"\nCode to analyze:\n{code_snippet}")
    print("\n--- Using KIE with Streaming for Real-time Feedback ---\n")

    try:
        async for chunk in client.chat(
            function_name="code_analysis",
            variant_name="kie_code_streaming",
            messages=[
                {
                    "role": "user",
                    "content": f"Please analyze this code and identify any issues:\n{code_snippet}",
                }
            ],
            stream=True,
        ):
            if chunk.text_delta:
                print(chunk.text_delta, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error during streaming: {e}")


async def example_compare_models():
    """Example: Compare KIE reasoning with standard model."""
    client = InferenceClient()

    prompt = "Explain how machine learning differs from deep learning in 2-3 sentences."

    print("=" * 80)
    print("Example 4: Comparing KIE with GPT-4 Turbo")
    print("=" * 80)
    print(f"\nPrompt: {prompt}")

    print("\n--- KIE with Reasoning ---")
    try:
        response = await client.chat(
            function_name="solve_problem",
            variant_name="kie_reasoning",
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- GPT-4 Turbo (for comparison) ---")
    try:
        response = await client.chat(
            function_name="solve_problem",
            variant_name="gpt_4_turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")

    print("\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("TensorZero KIE Reasoning Examples")
    print("=" * 80)
    print("\nThese examples demonstrate various uses of the KIE provider with TensorZero.")
    print("KIE supports:")
    print("  - Extended reasoning with configurable reasoning_effort")
    print("  - Streaming responses for real-time feedback")
    print("  - Including thought/reasoning content in responses")
    print("  - Tool calling and function calling capabilities")
    print("\n")

    # Run examples sequentially
    await example_solve_problem()
    await example_math_reasoning()
    await example_code_analysis()
    await example_compare_models()

    print("=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Note: Make sure your KIE_API_KEY environment variable is set
    # and the TensorZero gateway is running before executing this script
    asyncio.run(main())
