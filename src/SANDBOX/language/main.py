from transformers import pipeline

# Specify the model
model_name = "distilbert-base-uncased-distilled-squad"

# Create the QA pipeline
qa_pipeline = pipeline("question-answering", model=model_name)

def is_greeting(input_text):
    greetings = ["hello", "hi", "hey", "greetings", "hye", "hola"]
    return input_text.lower() in greetings

def ask_question(context, question):
    if not question.strip():
        return "Please ask a question."
    if is_greeting(question):
        return "Hello! How can I help you today?"

    result = qa_pipeline(question=question, context=context)
    
    # Check if the model provides a relevant answer
    if result['score'] < 0.1 or len(result['answer'].split()) < 3:
        return "I'm not sure about that. Could you ask something related to the context?"
    
    return result['answer']

def load_context(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load the initial context from a text file
context_file = "SANDBOX/language/context.txt"
context = load_context(context_file)

# Interactive loop to ask questions
conversation_history = context

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Goodbye!")
        break

    # Update the context with the user's question
    conversation_history += f"\nUser: {question}"
    
    answer = ask_question(conversation_history, question)
    
    # Update the context with the chatbot's answer
    conversation_history += f"\nChatbot: {answer}"
    
    print("Answer:", answer)
