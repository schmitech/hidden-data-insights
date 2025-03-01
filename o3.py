import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text(prompt: str, max_tokens: int = 50):
    try:
        # Check if API key is available
        if not openai.api_key:
            return "Error: OpenAI API key not found in environment variables."
            
        # Use the ChatCompletion API with the o3-mini model.
        response = openai.ChatCompletion.create(
            model="o3-mini",  # Specify the o3-mini model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        # Extract and return the generated text from the assistant's reply
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Example prompt
    prompt_text = "Explain the benefits of using renewable energy sources."
    output_text = generate_text(prompt_text, max_tokens=100)
    print("Generated Text:\n", output_text)
