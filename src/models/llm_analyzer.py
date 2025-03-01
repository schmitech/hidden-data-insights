import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMAnalyzer:
    """
    Uses OpenAI's API to analyze datasets and extract hidden patterns and insights.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the LLM analyzer with an OpenAI API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to load from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def _prepare_data_summary(self, df, max_rows=10):
        """
        Prepare a summary of the dataset for the LLM.
        
        Args:
            df (pd.DataFrame): The dataset to summarize
            max_rows (int): Maximum number of sample rows to include
            
        Returns:
            str: A text summary of the dataset
        """
        # Basic dataset info
        num_rows, num_cols = df.shape
        column_info = []
        
        for col in df.columns:
            dtype = df[col].dtype
            num_nulls = df[col].isna().sum()
            
            if pd.api.types.is_numeric_dtype(dtype):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                col_info = (
                    f"Column: {col} (numeric)\n"
                    f"  - Type: {dtype}\n"
                    f"  - Range: {min_val} to {max_val}\n"
                    f"  - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}\n"
                    f"  - Missing values: {num_nulls} ({num_nulls/num_rows*100:.1f}%)"
                )
            
            elif pd.api.types.is_datetime64_dtype(dtype):
                min_date = df[col].min()
                max_date = df[col].max()
                
                col_info = (
                    f"Column: {col} (datetime)\n"
                    f"  - Type: {dtype}\n"
                    f"  - Range: {min_date} to {max_date}\n"
                    f"  - Missing values: {num_nulls} ({num_nulls/num_rows*100:.1f}%)"
                )
            
            else:  # Categorical or object
                num_unique = df[col].nunique()
                most_common = df[col].value_counts().head(3).to_dict()
                
                col_info = (
                    f"Column: {col} (categorical/text)\n"
                    f"  - Type: {dtype}\n"
                    f"  - Unique values: {num_unique}\n"
                    f"  - Most common values: {most_common}\n"
                    f"  - Missing values: {num_nulls} ({num_nulls/num_rows*100:.1f}%)"
                )
            
            column_info.append(col_info)
        
        # Sample rows
        sample_rows = df.head(max_rows).to_string(index=False)
        
        # Correlations for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_info = ""
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Get top 10 correlations
            corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corrs.append((
                        numeric_cols[i], 
                        numeric_cols[j], 
                        corr_matrix.iloc[i, j]
                    ))
            
            # Sort by absolute correlation value
            corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Format top correlations
            top_corrs = [f"{a} and {b}: {c:.3f}" for a, b, c in corrs[:10]]
            correlation_info = "Top correlations:\n" + "\n".join(top_corrs)
        
        # Combine all information
        data_summary = (
            f"Dataset Summary:\n"
            f"- Dimensions: {num_rows} rows × {num_cols} columns\n\n"
            f"Column Information:\n" + "\n".join(column_info) + "\n\n"
            f"Sample Data (first {max_rows} rows):\n{sample_rows}\n\n"
            f"{correlation_info}"
        )
        
        return data_summary
    
    def analyze_dataset(self, df, domain=None, specific_questions=None):
        """
        Analyze a dataset using OpenAI's API to extract hidden patterns and insights.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
            domain (str, optional): The domain of the data (e.g., "e-commerce", "healthcare")
            specific_questions (list, optional): Specific questions to ask about the data
            
        Returns:
            dict: Analysis results including patterns, insights, and recommendations
        """
        # Prepare data summary
        data_summary = self._prepare_data_summary(df)
        
        # Construct prompt
        domain_context = f"This is a dataset from the {domain} domain. " if domain else ""
        
        specific_questions_text = ""
        if specific_questions:
            specific_questions_text = "Additionally, please answer these specific questions with ONLY direct, concise responses:\n"
            for i, question in enumerate(specific_questions, 1):
                specific_questions_text += f"{i}. {question}\n"
            specific_questions_text += "\nFor each specific question, start your answer with 'ANSWER TO QUESTION #X:' where X is the question number. Provide ONLY the direct answer without additional explanations, context, or follow-up commentary. Keep answers brief and to the point."
        
        prompt = f"""
        {domain_context}I'm going to provide you with a dataset summary. Your task is to analyze this data and identify hidden patterns, correlations, and insights that might not be immediately obvious.

        {data_summary}

        Please analyze this data and provide:
        1. A summary of the key characteristics of the dataset
        2. Identification of any hidden patterns or relationships you can detect
        3. Insights about unusual or unexpected correlations
        4. Potential causal relationships that might explain the patterns
        5. Recommendations for further analysis or actions based on these insights

        {specific_questions_text}

        Focus on finding non-obvious patterns that would be valuable for decision-making. Be specific and explain your reasoning.
        """
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analysis expert specializing in finding hidden patterns and insights in complex datasets. Your analysis should be thorough and insightful for general patterns, but extremely concise and direct when answering specific questions. For specific questions, provide only the exact answer without elaboration or context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        # Extract and structure the response
        analysis_text = response.choices[0].message.content
        
        # Parse the analysis into structured sections
        sections = {
            "summary": "",
            "hidden_patterns": [],
            "unusual_correlations": [],
            "causal_relationships": [],
            "recommendations": [],
            "specific_answers": []
        }
        
        # Simple parsing logic - can be improved for production
        current_section = "summary"
        
        # First pass: extract specific question answers
        if specific_questions:
            answer_pattern = r'ANSWER TO QUESTION #(\d+):(.*?)(?=ANSWER TO QUESTION #\d+:|$)'
            import re
            answers = re.findall(answer_pattern, analysis_text, re.DOTALL)
            
            for question_num, answer in answers:
                question_idx = int(question_num) - 1
                if question_idx < len(specific_questions):
                    question = specific_questions[question_idx]
                    # Clean the answer to remove any trailing explanatory text
                    # Look for common patterns that indicate the start of explanations
                    clean_answer = answer.strip()
                    explanation_starters = [
                        "This indicates", "This suggests", "This shows", 
                        "This means", "This is because", "By analyzing", 
                        "Looking at", "Based on", "In conclusion", "Therefore"
                    ]
                    
                    for starter in explanation_starters:
                        if starter in clean_answer:
                            # Split at the explanation starter and keep only the first part
                            parts = clean_answer.split(starter, 1)
                            clean_answer = parts[0].strip()
                    
                    # Also split on common sentence endings if they're followed by explanatory text
                    clean_answer = re.sub(r'(\.\s+)(?=[A-Z])', r'.\n', clean_answer)
                    # Take only the first sentence if it's a complete answer
                    sentences = clean_answer.split('\n')
                    if len(sentences) > 1 and len(sentences[0]) > 10:  # Ensure first sentence is substantial
                        clean_answer = sentences[0].strip()
                    
                    formatted_answer = f"Q: {question}\nA: {clean_answer}"
                    sections["specific_answers"].append(formatted_answer)
        
        # Second pass: extract other sections
        for line in analysis_text.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # Skip lines that are already processed as specific answers
            if "ANSWER TO QUESTION #" in line:
                continue
                
            if "hidden pattern" in line.lower() or "pattern" in line.lower() and "#" in line:
                current_section = "hidden_patterns"
                sections[current_section].append(line)
            elif "correlation" in line.lower() and "#" in line:
                current_section = "unusual_correlations"
                sections[current_section].append(line)
            elif "causal" in line.lower() and "#" in line:
                current_section = "causal_relationships"
                sections[current_section].append(line)
            elif "recommend" in line.lower() and "#" in line:
                current_section = "recommendations"
                sections[current_section].append(line)
            else:
                if current_section == "summary" and not sections[current_section]:
                    sections[current_section] = line
                else:
                    if isinstance(sections[current_section], list):
                        if sections[current_section]:
                            sections[current_section][-1] += " " + line
                        else:
                            sections[current_section].append(line)
                    else:
                        sections[current_section] += " " + line
        
        # Add the full raw analysis
        sections["raw_analysis"] = analysis_text
        
        return sections
    
    def compare_analyses(self, actual_patterns, discovered_patterns):
        """
        Compare the actual hidden patterns with those discovered by the LLM.
        
        Args:
            actual_patterns (list): List of actual hidden patterns in the data
            discovered_patterns (list): List of patterns discovered by the LLM
            
        Returns:
            dict: Comparison results including accuracy metrics
        """
        prompt = f"""
        I'm going to provide you with two lists:
        1. The actual hidden patterns that exist in a dataset
        2. The patterns discovered by an AI analysis

        Your task is to evaluate how well the AI discovered the actual patterns.

        Actual Hidden Patterns:
        {json.dumps(actual_patterns, indent=2)}

        Discovered Patterns:
        {json.dumps(discovered_patterns, indent=2)}

        Please analyze and provide:
        1. For each actual pattern, whether it was fully discovered, partially discovered, or missed
        2. Any false positives (patterns identified that don't actually exist)
        3. An overall accuracy score (0-100%)
        4. Suggestions for improving the analysis

        Be specific in your evaluation and explain your reasoning.
        """
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of data analysis results. Your job is to objectively assess how well an AI system has discovered hidden patterns in data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        # Return the evaluation
        return {
            "evaluation": response.choices[0].message.content,
            "actual_patterns": actual_patterns,
            "discovered_patterns": discovered_patterns
        } 