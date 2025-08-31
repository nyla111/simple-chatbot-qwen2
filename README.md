# Qwen Chatbot API Deployment

A complete solution for deploying the Qwen language model as a chatbot API with a web interface, using FastAPI and ngrok for public access.

## Overview

This project packages the Qwen/Qwen2-0.5B-Instruct model into a RESTful API using FastAPI, exposes it publicly via ngrok, and provides a web-based chat interface. It's perfect for demonstrations, testing, and sharing your AI chatbot with others.

## Features

- **FastAPI Backend**: Robust and scalable API server
- **Qwen2.5-3B-Instruct**: Lightweight yet powerful language model
- **Web Interface**: Beautiful, responsive chat UI
- **Public Access**: ngrok integration for instant public sharing
- **CORS Enabled**: Cross-origin requests supported
- **Conversation History**: Maintains context across messages

## Prerequisites

- Python 3.8+
- pip package manager
- Hugging Face account (to accept model terms)
- ngrok account (free tier available)

## Installation

1. **Clone the repository**

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install torch transformers accelerate uvicorn fastapi python-multipart
   ```

4. **Install ngrok**
   - Download from [ngrok.com](https://ngrok.com/)
   - Add to PATH or place in project directory
   - Authenticate with your token:
     ```bash
     ngrok authtoken YOUR_AUTH_TOKEN_HERE
     ```

## Quick Start

1. **Start the API server**
   ```bash
   python main.py
   ```
   Server starts at `http://localhost:8000`

2. **Test the API locally**
   ```bash
   python test_model.py
   ```
   Or use curl:
   ```bash
   curl -X POST "http://localhost:8000/chat/" \
   -H "Content-Type: application/json" \
   -d '{"message": "Hello, what can you do?", "history": []}'
   ```

3. **Expose with ngrok** (in new terminal)
   ```bash
   ngrok http 8000
   ```
   Copy the public URL (e.g., `https://abc123.ngrok-free.app`)

4. **Access the web interface**
   - Open `http://localhost:8000` for local access
   - Or use your ngrok URL for public access

## API Endpoints

### `GET /`
Returns the web chat interface.

### `POST /chat/`
Main chat endpoint.

**Request Body:**
```json
{
  "message": "user input",
  "history": [
    ["user message 1", "bot response 1"],
    ["user message 2", "bot response 2"]
  ]
}
```

**Response:**
```json
{
  "response": "bot response",
  "history": [
    ["user message 1", "bot response 1"],
    ["user message 2", "bot response 2"],
    ["current message", "current response"]
  ]
}
```

## Configuration

### Model Options
The project uses Qwen2-0.5B-Instruct by default. To use other models:

1. Modify `model_name` in `model_loader.py`:
   ```python
   # For larger model (requires more GPU memory)
   model_name = "Qwen/Qwen2-7B-Instruct"
   
   # For different model family
   model_name = "microsoft/phi-2"
   ```

2. Adjust generation parameters in `main.py`:
   ```python
   outputs = model.generate(
       **inputs,
       max_new_tokens=512,      # Response length
       temperature=0.7,         # Creativity (0-1)
       do_sample=True,
       top_p=0.9,               # Diversity
   )
   ```

## Troubleshooting

### Common Issues

1. **"Internal Server Error"**
   - Check if all dependencies are installed
   - Verify model downloads correctly

2. **CORS Errors**
   - Ensure CORS middleware is enabled in `main.py`

3. **ngrok Connection Issues**
   - Verify ngrok is authenticated
   - Check firewall settings

4. **Model Loading Failures**
   - Accept model terms on Hugging Face
   - Ensure sufficient disk space (~1GB for 0.5B model)

### Performance Tips

- Use GPU for faster inference (automatically detected)
- Reduce `max_new_tokens` for quicker responses
- Lower `temperature` for more deterministic outputs

## Usage Examples

### Python Client
```python
import requests

api_url = "https://your-ngrok-url.ngrok-free.app/chat/"

response = requests.post(api_url, json={
    "message": "What is AI?",
    "history": []
})

print(response.json()["response"])
```

### JavaScript Client
```javascript
const response = await fetch('https://your-ngrok-url.ngrok-free.app/chat/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: 'Hello!',
        history: []
    })
});
const data = await response.json();
console.log(data.response);
```

## Security Notes

- This setup is for **development/demonstration purposes only**
- ngrok free tier URLs are temporary and change on restart
- For production use, consider:
  - Proper authentication
  - Rate limiting
  - HTTPS encryption
  - Cloud deployment (AWS, GCP, Azure)

## Scaling Options

For higher traffic or larger models:

1. **Upgrade Hardware**: Use better GPU with more VRAM
2. **Cloud Deployment**: Deploy to cloud platforms
3. **Load Balancing**: Use multiple ngrok instances
4. **Model Optimization**: Use quantization for smaller models

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project uses the Qwen model which has its own license terms. Please check the [Hugging Face model card](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [ngrok](https://ngrok.com/) for tunneling service
- [Qwen team](https://github.com/QwenLM) for the language models

---

**Happy Chatbot Building!** 🚀

For questions or support, please open an issue in the GitHub repository.
