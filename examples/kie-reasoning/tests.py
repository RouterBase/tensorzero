#!/usr/bin/env python3
"""
KIE Provider Test Suite - Consolidated Tests
Covers basic inference, models, media, and advanced features
"""

import json
import requests
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass

BASE_URL = "http://localhost:3000"
INFERENCE_ENDPOINT = f"{BASE_URL}/inference"

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")

def check_server():
    """Check if gateway is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def format_response(data: Dict[str, Any]) -> str:
    """Format response for display"""
    return json.dumps(data, indent=2)[:300] + "..." if len(json.dumps(data)) > 300 else json.dumps(data, indent=2)

# ============================================================================
# BASIC INFERENCE TESTS
# ============================================================================

def test_basic_high_reasoning() -> TestResult:
    """Test basic inference with high reasoning effort"""
    print_section("Test 1: Basic Inference - High Reasoning Effort")
    
    payload = {
        "model_name": "kie::gemini-3-pro",
        "input": {
            "messages": [
                {"role": "user", "content": "What is the square root of 144?"}
            ]
        },
        "params": {
            "chat_completion": {
                "max_tokens": 500,
                "reasoning_effort": "high",
                "temperature": 0.7
            }
        },
        "stream": False
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Response received{RESET}")
        return TestResult("Basic High Reasoning", True, f"Status {response.status_code}")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Basic High Reasoning", False, str(e))

def test_basic_low_reasoning() -> TestResult:
    """Test basic inference with low reasoning effort"""
    print_section("Test 2: Basic Inference - Low Reasoning Effort")
    
    payload = {
        "model_name": "kie::gemini-3-pro",
        "input": {
            "messages": [
                {"role": "user", "content": "List 3 colors"}
            ]
        },
        "params": {
            "chat_completion": {
                "max_tokens": 300,
                "reasoning_effort": "low",
                "temperature": 0.5
            }
        },
        "stream": False
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Response received{RESET}")
        return TestResult("Basic Low Reasoning", True, f"Status {response.status_code}")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Basic Low Reasoning", False, str(e))

# ============================================================================
# MODEL VARIANT TESTS
# ============================================================================

def test_model_variant(model_name: str, variant_name: str) -> TestResult:
    """Test a specific model variant"""
    print_section(f"Testing Model: {variant_name}")
    
    payload = {
        "model_name": f"kie::{model_name}",
        "input": {
            "messages": [
                {"role": "user", "content": "What makes a good language? (Brief)"}
            ]
        },
        "params": {
            "chat_completion": {
                "max_tokens": 300,
                "reasoning_effort": "low",
                "temperature": 0.5
            }
        },
        "stream": False
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Model {variant_name} works{RESET}")
        return TestResult(f"Model: {variant_name}", True, f"Status {response.status_code}")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult(f"Model: {variant_name}", False, str(e))

def test_all_models() -> list:
    """Test all available model variants"""
    print_section("All Model Variants Test")
    
    models = [
        ("gemini-3-pro", "Gemini 3 Pro"),
        ("gemini-3-flash", "Gemini 3 Flash"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ]
    
    results = []
    for model, name in models:
        result = test_model_variant(model, name)
        results.append(result)
    return results

# ============================================================================
# RICH MEDIA TESTS
# ============================================================================

def test_image_analysis() -> TestResult:
    """Test image analysis with media file"""
    print_section("Test: Image Analysis")
    
    payload = {
        "function_name": "image_analysis",
        "variant_name": "kie_vision",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "file",
                            "url": "https://file.aiquickdraw.com/custom-page/akr/section-images/1759055072437dqlsclj2.png",
                            "mime_type": "image/jpeg"
                        }
                    ]
                }
            ]
        },
        "params": {
            "chat_completion": {
                "reasoning_effort": "high"
            }
        },
        "stream": True
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=6000)
        print(f"response status code: {response.status_code}");
        response.raise_for_status()
        chunks = 0
        for line in response.iter_lines():
            if line:
                chunks += 1
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Received {chunks} streaming chunks{RESET}")
        return TestResult("Image Analysis", True, f"Received {chunks} chunks")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Image Analysis", False, str(e))

def test_video_analysis() -> TestResult:
    """Test video analysis with media file"""
    print_section("Test: Video Analysis")
    
    payload = {
        "function_name": "video_analysis",
        "variant_name": "kie_video",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video"},
                        {
                            "type": "file",
                            "url": "https://media.w3.org/2010/05/sintel/trailer.mp4",
                            "mime_type": "video/mp4"
                        }
                    ]
                }
            ]
        },
        "stream": True 
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=90)
        response.raise_for_status()
        chunks = 0
        for line in response.iter_lines():
            if line:
                print(line)
                chunks += 1 
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Video analysis works{RESET}")
        return TestResult("Video Analysis", True, f"Received {chunks} chunks")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Video Analysis", False, str(e))

def test_pdf_analysis() -> TestResult:
    """Test PDF document analysis"""
    print_section("Test: PDF Document Analysis")
    
    payload = {
        "function_name": "document_analysis",
        "variant_name": "kie_document",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this document?"},
                        {
                            "type": "file",
                            "url": "https://eprint.iacr.org/2026/049.pdf",
                            "mime_type": "application/pdf"
                        }
                    ]
                }
            ]
        },
        "stream": True
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        chunks = 0
        for line in response.iter_lines():
            if line:
                print(line)
                chunks += 1

        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ PDF analysis works{RESET}")
        return TestResult("PDF Analysis", True, f"Received {chunks} chunks")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("PDF Analysis", False, str(e))

def test_multimodal() -> TestResult:
    """Test multimodal reasoning with multiple media"""
    print_section("Test: Multimodal Reasoning")
    
    payload = {
        "function_name": "multimodal_reasoning",
        "variant_name": "kie_multimodal",
        "input": {
            "system": "You are an expert analyst.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images"},
                        {
                            "type": "file",
                            "url": "https://file.aiquickdraw.com/custom-page/akr/section-images/1759055072437dqlsclj2.png",
                            "mime_type": "image/jpeg"
                        },
                        {
                            "type": "file",
                            "url": "https://pixnio.com/free-images/2017/03/01/2017-03-01-16-28-35-960x640.jpg",
                            "mime_type": "image/jpeg"
                        }
                    ]
                }
            ]
        },
        "stream": True 
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()

        chunks = 0
        for line in response.iter_lines():
            if line:
                print(line)
                chunks += 1

        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Multimodal reasoning works{RESET}")
        return TestResult("Multimodal Reasoning", True, f"Received {chunks} chunks")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Multimodal Reasoning", False, str(e))

# ============================================================================
# ADVANCED TESTS
# ============================================================================

def test_function_variant() -> TestResult:
    """Test using function with variant"""
    print_section("Test: Function Variant")
    
    payload = {
        "function_name": "image_analysis",
        "variant_name": "kie_vision",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "file",
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
                            "mime_type": "image/jpeg"
                        }
                    ]
                }
            ]
        }
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Function variant works{RESET}")
        return TestResult("Function Variant", True, f"Status {response.status_code}")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Function Variant", False, str(e))

def test_streaming() -> TestResult:
    """Test streaming response"""
    print_section("Test: Streaming Response")
    
    payload = {
        "model_name": "kie::gemini-3-pro",
        "input": {
            "messages": [
                {"role": "user", "content": "Write a short poem"}
            ]
        },
        "params": {
            "chat_completion": {
                "max_tokens": 300,
                "reasoning_effort": "low"
            }
        },
        "stream": True
    }
    
    try:
        response = requests.post(INFERENCE_ENDPOINT, json=payload, timeout=60, stream=True)
        response.raise_for_status()
        chunks = 0
        for line in response.iter_lines():
            if line:
                chunks += 1
        print(f"{GREEN}✓ Status: {response.status_code}{RESET}")
        print(f"{GREEN}✓ Received {chunks} streaming chunks{RESET}")
        return TestResult("Streaming", True, f"Received {chunks} chunks")
    except Exception as e:
        print(f"{RED}✗ Error: {str(e)}{RESET}")
        return TestResult("Streaming", False, str(e))

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests"""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  KIE PROVIDER TEST SUITE{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    
    # Check if server is running
    if not check_server():
        print(f"\n{RED}✗ Gateway is not running!{RESET}")
        print(f"  Start the gateway with:")
        print(f"  {YELLOW}RUST_LOG=debug cargo run -r --bin gateway -- --config-file examples/kie-reasoning/config/tensorzero.toml{RESET}")
        sys.exit(1)
    
    print(f"\n{GREEN}✓ Gateway is running{RESET}\n")
    
    results = []
    
    # # Basic tests
    # print(f"\n{BOLD}SECTION 1: BASIC INFERENCE{RESET}")
    # results.append(test_basic_high_reasoning())
    # results.append(test_basic_low_reasoning())
    
    # # Model tests
    # print(f"\n{BOLD}SECTION 2: MODEL VARIANTS{RESET}")
    # results.extend(test_all_models())
    
    # Media tests
    print(f"\n{BOLD}SECTION 3: RICH MEDIA (Images, Videos, PDFs){RESET}")
    #results.append(test_image_analysis())
    #results.append(test_video_analysis())
    #results.append(test_pdf_analysis())
    results.append(test_multimodal())
    
    # # Advanced tests
    # print(f"\n{BOLD}SECTION 4: ADVANCED FEATURES{RESET}")
    # results.append(test_function_variant())
    # results.append(test_streaming())
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result.passed else f"{RED}✗ FAIL{RESET}"
        print(f"{status}: {result.name}")
    
    print(f"\n{BOLD}Total: {passed}/{total} tests passed{RESET}\n")
    
    if passed == total:
        print(f"{GREEN}✓ ALL TESTS PASSED{RESET}\n")
        return 0
    else:
        print(f"{RED}✗ {total - passed} TEST(S) FAILED{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
