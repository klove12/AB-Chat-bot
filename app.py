from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch

# Load GPT-Neo model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def get_chat_response(text):
    # Tokenize the input text and add a batch dimension
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Create attention mask (1 for tokens that are not padding, 0 for padding tokens)
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # Generate a response with attention mask
    chat_history_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response and return it
    return tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

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
