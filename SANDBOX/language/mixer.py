import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Structured data containing recommendations for nodes
ARTICLE = """

"""

def generate_response(prompt, max_length=32):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def find_recommendations(node_name):
    for line in ARTICLE.strip().split("\n"):
        if node_name in line.lower():
            return line
    return None

print("Chatbot: Hi there! Let's talk about recommended nodes. What would you like to know?")
while True:
    user_input = input("You: ").strip().lower()

    if user_input == "exit":
        print("Chatbot: Goodbye!")
        break

    # Normalize user input
    normalized_input = user_input.replace("recommendations for ", "").strip()

    # Find recommendations in the article
    recommendation = find_recommendations(normalized_input)
    if recommendation:
        print("Chatbot:", recommendation)
    else:
        # Generate a response using GPT-2
        response = generate_response(f"Tell me about recommendations for the node {normalized_input}")
        print("Chatbot:", response)
