# test_small.py
from model_loader import model, tokenizer  # Remove device import since we're using device_map="auto"
import torch

def test_small_model():
    print("🤖 Testing small model...")
    
    try:
        # Simple chat format
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"Formatted input: {text}")
        
        # Get device from model
        device = model.device
        print(f"Using device: {device}")
        
        # Tokenize and generate
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ SUCCESS! Response: {response}")
        return True
        
    except Exception as e:
        print(f"💥 FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_small_model()