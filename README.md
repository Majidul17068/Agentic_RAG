# SopAssist AI - Bilingual Policy Assistant

An AI-powered SOP and Policy Assistant with full Bengali language support, built for internal teams to ask questions about company policies stored as images and PDFs.

## 🌟 Features

- **Bilingual Support**: Full English and Bengali language support
- **OCR Processing**: Extract text from images with Bengali OCR support
- **PDF Processing**: Process PDF policy documents with automatic language detection
- **Vector Search**: Semantic search using ChromaDB
- **Free LLM Integration**: Uses Groq and Hugging Face free models
- **Agentic RAG**: Multi-agent system for comprehensive policy analysis
- **Modern UI**: Beautiful Streamlit frontend
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Local Deployment**: Everything runs locally, no paid APIs required

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd sopassist-ai
```

### 2. Environment Setup

Create a `.env` file with your configuration:

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your settings
# Your Groq API key is already included in the example
```

The `.env` file contains:
```bash
# LLM Provider (groq, huggingface, ollama)
LLM_PROVIDER=groq

# Your Groq API Key (already configured)
GROQ_API_KEY=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

# API Ports
API_PORT=8000
STREAMLIT_PORT=8501
```

### 3. Install Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ben python3-pip

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Setup Bengali Support

```bash
# Run the Bengali setup script
python scripts/setup_bengali_support.py
```

### 5. Start the Application

```bash
# Start all services
python scripts/startup.py
```

The application will be available at:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
sopassist-ai/
├── backend/                 # FastAPI backend
├── core/                   # Core processing modules
│   ├── ocr_processor.py    # OCR with Bengali support
│   ├── pdf_processor.py    # PDF processing with Bengali support
│   ├── vector_store.py     # ChromaDB vector store
│   ├── free_llm_interface.py # Free LLM integration
│   └── policy_agents.py    # Agentic RAG agents
├── frontend/               # Streamlit frontend
├── config/                 # Configuration files
│   └── settings.py         # Main settings (uses .env)
├── scripts/                # Utility scripts
├── data/                   # Data storage
├── logs/                   # Application logs
├── Policy file/            # PDF policy files (create this folder)
├── .env                    # Environment variables (create from env.example)
├── env.example             # Example environment file
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

### Environment Variables (.env file)

The system uses a `.env` file for all configuration. Key settings:

```bash
# LLM Provider (groq, huggingface, ollama)
LLM_PROVIDER=groq

# Your Groq API Key
GROQ_API_KEY=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

# API Ports
API_PORT=8000
STREAMLIT_PORT=8501

# Bengali Language Support
BENGALI_DETECTION_THRESHOLD=0.1
BENGALI_OCR_LANG=ben+eng
```

### Bengali Language Support

Bengali support is automatically enabled with:
- Bengali OCR (`tesseract-ocr-ben`)
- Bengali text detection
- Bilingual responses
- Bengali policy categorization

## 📄 Adding Policy Documents

### Method 1: Upload via Web Interface

1. Start the application
2. Go to http://localhost:8501
3. Use the file upload interface to add images or PDFs

### Method 2: Process PDF Folder

1. Create a `Policy file` folder in the project root
2. Add your PDF policy files (named by policy type)
3. Run the processing script:

```bash
python scripts/process_pdf_policies.py
```

### Supported File Types

- **Images**: JPG, JPEG, PNG, TIFF, BMP
- **Documents**: PDF (with Bengali support)

## 🤖 Using the System

### Web Interface

1. **Upload Documents**: Use the file upload interface
2. **Ask Questions**: Type questions in English or Bengali
3. **View Results**: Get detailed answers with source citations

### API Usage

```python
import requests

# Query policies
response = requests.post("http://localhost:8000/query", json={
    "question": "What is the vacation policy?",
    "top_k": 5
})

# Process PDF
with open("policy.pdf", "rb") as f:
    response = requests.post("http://localhost:8000/process-pdf", files={"file": f})
```

### Bengali Queries

The system automatically detects Bengali text and responds accordingly:

```python
# Bengali question
response = requests.post("http://localhost:8000/query", json={
    "question": "ছুটির নীতি কী?",
    "top_k": 5
})
```

## 🔍 Agentic RAG System

The system uses multiple specialized agents:

1. **Research Agent**: Finds relevant policy information
2. **Analysis Agent**: Analyzes and interprets policies
3. **Synthesis Agent**: Combines information into comprehensive answers
4. **Communication Agent**: Formats responses appropriately

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Features

1. Core modules are in `core/`
2. API endpoints in `backend/main.py`
3. Frontend components in `frontend/app.py`

### Logging

Logs are stored in `logs/` with different levels:
- `app.log`: General application logs
- `error.log`: Error logs
- `startup.log`: Startup process logs

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**:
   ```bash
   sudo apt-get install tesseract-ocr tesseract-ocr-ben
   ```

2. **Bengali OCR not working**:
   ```bash
   python scripts/setup_bengali_support.py
   ```

3. **API key issues**:
   - Check your `.env` file
   - Verify Groq API key is valid
   - Ensure `.env` file is in the project root

4. **Port conflicts**:
   - Change ports in `.env` file
   - Check if ports 8000/8501 are available

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python scripts/startup.py
```

## 📊 Performance

- **OCR Processing**: ~2-5 seconds per image
- **PDF Processing**: ~1-3 seconds per page
- **Query Response**: ~1-2 seconds
- **Bengali Support**: No performance impact

## 🔒 Security

- All processing is local
- No data sent to external services (except LLM APIs)
- API keys stored in `.env` file (not committed to git)
- CORS enabled for local development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/`
3. Open an issue on GitHub

---

**SopAssist AI** - Making company policies accessible in any language! 🌍 