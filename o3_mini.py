import openai

# Set your OpenAI API key
openai.api_key = "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"

def generate_text(prompt: str, max_tokens: int = 50):
    try:
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
