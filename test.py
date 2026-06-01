from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=30):
    # Load the fine-tuned model and tokenizer from the saved folder
    model_path = "./my_finetuned_gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Encode the input prompt into tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text sequences based on the prompt
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode tokens back into a readable string
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    print("\n--- Testing Our Fine-Tuned GPT-2 Model ---")
    
    # Define your testing prompt
    prompt_text = "Artificial intelligence is"
    
    print(f"Input Prompt: '{prompt_text}'")
    print("Model is generating text, please wait...")
    
    result = generate_text(prompt_text)
    print(f"\nGenerated Output:\n{result}\n")
    print("-" * 40)