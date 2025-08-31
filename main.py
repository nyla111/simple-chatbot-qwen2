# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from model_loader import load_model
model, tokenizer = load_model()
import uvicorn
import torch

# Create the app
app = FastAPI(title="Qwen2.5-3B-Instruct")

# Add CORS middleware
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

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chatbox { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .user { color: blue; text-align: right; margin: 5px 0; }
        .bot { color: green; margin: 5px 0; }
        .loading { color: gray; font-style: italic; }
        #userInput { width: 70%; padding: 10px; margin-right: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>Chat with AI Assistant</h1>
    <div id="chatbox"></div>
    <div>
        <input type="text" id="userInput" placeholder="Type your message here...">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        const API_URL = "/chat/";
        
        let conversationHistory = [];

        function addMessage(sender, text, isLoading = false) {
            const chatbox = document.getElementById('chatbox');
            const messageElement = document.createElement('div');
            messageElement.classList.add(sender);
            if (isLoading) {
                messageElement.classList.add('loading');
            }
            messageElement.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chatbox.appendChild(messageElement);
            chatbox.scrollTop = chatbox.scrollHeight;
        }

        async function sendMessage() {
            const userInput = document.getElementById('userInput');
            const message = userInput.value.trim();

            if (!message) return;

            addMessage('user', message);
            userInput.value = '';
            userInput.disabled = true;

            addMessage('bot', 'Thinking...', true);

            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        history: conversationHistory
                    })
                });

                // Remove loading
                const chatbox = document.getElementById('chatbox');
                const lastChild = chatbox.lastChild;
                if (lastChild.classList.contains('loading')) {
                    chatbox.removeChild(lastChild);
                }

                if (!response.ok) throw new Error('API error');
                
                const data = await response.json();
                addMessage('bot', data.response);
                conversationHistory = data.history;

            } catch (error) {
                console.error('Error:', error);
                const chatbox = document.getElementById('chatbox');
                const lastChild = chatbox.lastChild;
                if (lastChild.classList.contains('loading')) {
                    chatbox.removeChild(lastChild);
                }
                addMessage('bot', 'Sorry, connection error. Try again.');
            } finally {
                userInput.disabled = false;
                userInput.focus();
            }
        }

        document.getElementById('userInput').addEventListener('keypress', function (e) {
            if (e.key === 'Enter') sendMessage();
        });

        document.getElementById('userInput').focus();
        addMessage('bot', 'Hello! How can I help you today?');
    </script>
</body>
</html>
"""

# Serve HTML at root URL
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=HTML_CONTENT)

# API endpoint
@app.post("/chat/")
async def chat_with_model(request: ChatRequest):
    try:
        print(f"User asked: {request.message}")
        
        messages = []
        for user_msg, assistant_msg in request.history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": request.message})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"Model responded: {response}")
        updated_history = request.history + [[request.message, response]]
        
        return {"response": response, "history": updated_history}

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
