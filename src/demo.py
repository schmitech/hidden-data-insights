import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from utils.data_generator import DataGenerator
from models.llm_analyzer import LLMAnalyzer
from visualization.visualizer import DataVisualizer

# Load environment variables
load_dotenv()

def run_demo():
    """
    Run a demonstration of the Hidden Data Insights application.
    This shows how to use the components programmatically without the web interface.
    """
    print("=" * 80)
    print("Hidden Data Insights - Demonstration")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        print("You can add it to a .env file in the project root directory.")
        return
    
    # Initialize components
    data_generator = DataGenerator()
    llm_analyzer = LLMAnalyzer(api_key=api_key)
    visualizer = DataVisualizer()
    
    # Generate a dataset
    print("\nGenerating e-commerce dataset...")
    df = data_generator.generate_ecommerce_data(num_records=500)
    print(f"Generated dataset with {len(df)} records and {len(df.columns)} columns")
    
    # Display dataset info
    print("\nDataset Preview:")
    print(df.head())
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Define the actual patterns in the data
    actual_patterns = [
        "Seasonal trends with peaks on weekends",
        "Price sensitivity varies by customer segment",
        "Product category performance correlates with marketing spend",
        "Customer lifetime value is influenced by first purchase category",
        "Certain product combinations have higher than expected co-purchase rates"
    ]
    
    print("\nActual Hidden Patterns:")
    for i, pattern in enumerate(actual_patterns, 1):
        print(f"{i}. {pattern}")
    
    # Analyze the dataset
    print("\nAnalyzing dataset with o3-mini...")
    analysis_results = llm_analyzer.analyze_dataset(
        df,
        domain="e-commerce",
        specific_questions=[
            "Which customer segments are most profitable?",
            "What product categories show seasonal trends?",
            "Are there any unusual co-purchase patterns?"
        ]
    )
    
    # Display analysis results
    print("\nAnalysis Summary:")
    print(analysis_results["summary"])
    
    print("\nDiscovered Hidden Patterns:")
    for i, pattern in enumerate(analysis_results["hidden_patterns"], 1):
        print(f"{i}. {pattern}")
    
    print("\nUnusual Correlations:")
    for i, corr in enumerate(analysis_results["unusual_correlations"], 1):
        print(f"{i}. {corr}")
    
    print("\nPotential Causal Relationships:")
    for i, rel in enumerate(analysis_results["causal_relationships"], 1):
        print(f"{i}. {rel}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(analysis_results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Compare actual vs. discovered patterns
    print("\nComparing actual vs. discovered patterns...")
    comparison = llm_analyzer.compare_analyses(actual_patterns, analysis_results["hidden_patterns"])
    
    print("\nEvaluation:")
    print(comparison["evaluation"])
    
    # Save results to file
    print("\nSaving results to 'demo_results.txt'...")
    with open("demo_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Hidden Data Insights - Demonstration Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Actual Hidden Patterns:\n")
        for i, pattern in enumerate(actual_patterns, 1):
            f.write(f"{i}. {pattern}\n")
        
        f.write("\nDiscovered Hidden Patterns:\n")
        for i, pattern in enumerate(analysis_results["hidden_patterns"], 1):
            f.write(f"{i}. {pattern}\n")
        
        f.write("\nEvaluation:\n")
        f.write(comparison["evaluation"])
    
    print("\nDemonstration complete! Results saved to 'demo_results.txt'")
    print("=" * 80)
    print("To run the web application, execute: python src/app.py")
    print("=" * 80)

if __name__ == "__main__":
    run_demo() 