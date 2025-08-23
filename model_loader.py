# model_loader_small.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a much smaller model that will download faster
model_name = "Qwen/Qwen2-0.5B-Instruct"

print(f"Loading smaller model: {model_name}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()

print("✅ Small model and tokenizer loaded successfully!")