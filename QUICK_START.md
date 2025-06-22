# SopAssist AI - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites

1. **Python 3.9+**
2. **Tesseract OCR**
3. **Ollama with Llama3**

### Installation

1. **Install Tesseract OCR:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   ```

2. **Install Ollama and Llama3:**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull Llama3 model (in another terminal)
   ollama pull llama3
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Run the startup script:**
   ```bash
   python start_sopassist.py
   ```

2. **Choose option 1** to check dependencies

3. **Choose option 2** to setup directories

4. **Choose option 3** to add sample data

5. **Choose option 7** to start both API and frontend

6. **Open your browser** to:
   - Frontend: http://localhost:8501
   - API docs: http://localhost:8000/docs

### Test the System

1. Go to the "Ask Questions" page
2. Ask: "What is the vacation policy?"
3. You should get a detailed answer based on the sample policies

### Adding Your Own Policies

1. **Upload images** via the web interface
2. **Or** place images in `data/images/` and run:
   ```bash
   python scripts/process_images.py
   ```

### Manual Startup

If you prefer manual startup:

1. **Start API:**
   ```bash
   python api/main.py
   ```

2. **Start Frontend** (in another terminal):
   ```bash
   streamlit run frontend/app.py
   ```

### Troubleshooting

- **Tesseract not found**: Install Tesseract OCR
- **Ollama connection error**: Make sure Ollama is running and llama3 is pulled
- **Import errors**: Install missing packages with `pip install -r requirements.txt`

### Features

- ✅ **OCR Processing**: Extract text from policy images
- ✅ **Semantic Search**: Find relevant policies
- ✅ **AI Assistant**: Get intelligent answers
- ✅ **Web Interface**: User-friendly Streamlit app
- ✅ **API Backend**: Programmatic access
- ✅ **Agent Orchestration**: Advanced reasoning with CrewAI

### Next Steps

- Add your company's policy images
- Customize the configuration in `config/settings.py`
- Explore the API endpoints at http://localhost:8000/docs
- Run tests with `pytest tests/`

---

**Need help?** Check the full README.md for detailed documentation. 