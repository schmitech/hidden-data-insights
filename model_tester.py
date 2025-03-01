#!/usr/bin/env python3
"""
Model Tester Script for Hidden Data Insights

This script allows you to test different OpenAI models with the Hidden Data Insights application
without modifying your .env file. It temporarily sets the OPENAI_MODEL environment variable
for the duration of the script execution.

Usage:
    python model_tester.py --model MODEL_NAME [--demo] [--app]
    python model_tester.py --list

Examples:
    python model_tester.py --model gpt-4 --demo
    python model_tester.py --model gpt-3.5-turbo --app
    python model_tester.py --list
"""

import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv

# Available models to test
AVAILABLE_MODELS = [
    "gpt-4.5-preview",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test different OpenAI models with Hidden Data Insights")
    parser.add_argument("--model", type=str, help="OpenAI model to test")
    parser.add_argument("--demo", action="store_true", help="Run the demo script")
    parser.add_argument("--app", action="store_true", help="Run the web application")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        print("Available models to test:")
        for model in AVAILABLE_MODELS:
            print(f"  - {model}")
        return
    
    # Check if model is provided for non-list operations
    if not args.model:
        parser.error("the --model argument is required unless using --list")
    
    # Load environment variables from .env
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY in your .env file.")
        return
    
    # Set the model environment variable
    os.environ["OPENAI_MODEL"] = args.model
    print(f"Testing with model: {args.model}")
    
    # Run the requested script
    if args.demo:
        print("Running demo script...")
        subprocess.run(["python", "src/demo.py"])
    elif args.app:
        print("Running web application...")
        subprocess.run(["python", "src/app.py"])
    else:
        print("Please specify either --demo or --app to run a script.")
        print("Example: python model_tester.py --model gpt-4 --demo")

if __name__ == "__main__":
    main()
