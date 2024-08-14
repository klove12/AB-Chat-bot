from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

# History to maintain conversation context
chat_history_ids = None

def get_chat_response(text):
    global chat_history_ids

    # Tokenize input text and move to GPU
    new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt").to(device)

    # Concatenate with chat history
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Generate a response with increased temperature and beams
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.0,  # I set higher temperature for more diverse responses
        num_beams=5,      # I set more beams for better coherence
        no_repeat_ngram_size=2,  # I used this to avoid repeating n-grams
        early_stopping=True
    )
    
    # Update chat history
    chat_history_ids = outputs

    # Decode and return the response
    response = tokenizer.decode(outputs[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

def chat():
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        response = get_chat_response(user_input)
        print("Bot:", response)

if __name__ == '__main__':
    chat()
