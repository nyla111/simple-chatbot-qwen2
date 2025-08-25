from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import model, tokenizer
import uvicorn
import torch

app = FastAPI(title="Qwen2-0.5B Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = [] 

@app.get("/")
def read_root():
    return {"message": "Qwen2-0.5B Chat API is running!"}

@app.post("/chat/")
async def chat_with_model(request: ChatRequest):
    try:
        print(f"User asked: {request.message}")
        
        # Build messages in chat format
        messages = []
        
        # Add history if any
        for user_msg, assistant_msg in request.history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Add current message
        messages.append({"role": "user", "content": request.message})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        # Decode only the new tokens (response)
        response_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"Model responded: {response}")
        
        # Update history
        updated_history = request.history + [[request.message, response]]
        
        return {
            "response": response,
            "history": updated_history
        }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
