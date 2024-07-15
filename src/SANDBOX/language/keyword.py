# Define the structured data
ARTICLE = """
Recommended nodes for node vishesh : ['vinayak', 'priyanka', 'adhiraaj']
Recommended nodes for node shrestha: ['vishesh ']
Recommended nodes for node biswajeet: ['priyanka', 'vinayak', 'adhiraaj']
Recommended nodes for node priyanka: ['adhiraaj', 'yash']
Recommended nodes for node poonam: ['adhiraaj', 'yash', 'sachin']
Recommended nodes for node adhiraaj: ['vinayak', 'poonam', 'priyanka']
Recommended nodes for node yash: ['priyanka', 'adhiraaj', 'poonam']
Recommended nodes for node sachin: ['adhiraaj', 'kranti', 'poonam']
Recommended nodes for node vinayak: ['adhiraaj', 'biswajeet', 'kranti']
Recommended nodes for node kranti: ['vinayak', 'yash', 'adhiraaj']
Recommended nodes for node sai: ['adhiraaj', 'kranti', 'biswajeet']
"""

def generate_conversation(article):
    # Split the article into lines
    lines = article.strip().split('\n')
    
    # Initialize an empty conversation list
    conversation = []
    
    # Iterate through each line and construct conversation-like output
    for line in lines:
        node_info = line.split(':')
        node_name = node_info[0].strip()
        recommendations = node_info[1].strip()
        
        # Create a conversation-like structure
        conversation.append(f"You asked about node {node_name}.")
        conversation.append(f"Here are the recommended nodes: {recommendations}.")
    
    # Join all conversation lines into a single string
    conversation_text = '\n'.join(conversation)
    
    return conversation_text

# Generate the conversation text
conversation = generate_conversation(ARTICLE)

# Print the conversation
print(conversation)
