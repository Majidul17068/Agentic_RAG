# SopAssist AI

A complete, local AI-powered system for querying company SOP/HR/tech policies stored as images. Built with 100% local components - no paid APIs required.

## Features

- **Image Processing**: Extract text from policy images using Tesseract OCR
- **Semantic Search**: Find relevant policies using sentence embeddings
- **AI Assistant**: Get intelligent answers using local Llama3 model via Ollama
- **Web Interface**: User-friendly Streamlit frontend
- **API Backend**: FastAPI for programmatic access
- **Agent Orchestration**: CrewAI for complex multi-step queries

## Tech Stack

- **OCR**: Tesseract via `pytesseract`
- **Embeddings**: `sentence-transformers` model `all-MiniLM-L6-v2`
- **Vector DB**: `chromadb`
- **LLM**: Local Ollama model `llama3`
- **Agent Orchestration**: `crewai`
- **API**: `FastAPI`
- **Frontend**: `Streamlit`
- **Python**: ≥ 3.9

## Quick Start

### Prerequisites

1. **Install Tesseract OCR**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Install Ollama and Llama3**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama3 model
   ollama pull llama3
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. **Clone and setup**:
   ```bash
   cd sopassist-ai
   python setup.py
   ```

2. **Add policy images**:
   ```bash
   # Copy your policy images to data/images/
   cp your_policies/*.jpg data/images/
   cp your_policies/*.png data/images/
   ```

3. **Process images**:
   ```bash
   python scripts/process_images.py
   ```

### Run the Application

1. **Start the API backend**:
   ```bash
   python api/main.py
   ```

2. **Start the Streamlit frontend** (in a new terminal):
   ```bash
   streamlit run frontend/app.py
   ```

3. **Access the application**:
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - API docs: http://localhost:8000/docs

## Usage

### Web Interface
1. Open the Streamlit app in your browser
2. Type your question about company policies
3. Get instant answers with relevant policy references

### API Usage
```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", 
                        json={"question": "What is the vacation policy?"})
print(response.json())
```

## Project Structure

```
sopassist-ai/
├── api/                    # FastAPI backend
├── frontend/              # Streamlit frontend
├── core/                  # Core business logic
├── agents/                # CrewAI agents
├── data/                  # Data storage
├── scripts/               # Utility scripts
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── setup.py              # Setup script
└── README.md             # This file
```

## Configuration

Edit `config/settings.py` to customize:
- Vector database settings
- Model parameters
- API endpoints
- File paths

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Features
1. Create feature branch
2. Add tests in `tests/`
3. Update documentation
4. Submit pull request

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and in PATH
2. **Ollama connection error**: Make sure Ollama is running and llama3 model is pulled
3. **Memory issues**: Reduce batch size in config for large image collections

### Logs
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `logs/`
- Open an issue on GitHub 