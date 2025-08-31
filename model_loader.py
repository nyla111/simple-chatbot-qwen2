from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading smaller model: {model_name}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()

    print("Small model and tokenizer loaded successfully!")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
