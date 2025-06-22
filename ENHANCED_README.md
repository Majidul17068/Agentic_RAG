# 🎯 Enhanced SopAssist AI - Agentic RAG Policy Assistant

A comprehensive AI-powered policy assistant that processes PDF policy documents and provides intelligent answers using **free LLM models** (Groq, Hugging Face) and **agentic RAG** with CrewAI.

## 🚀 Key Features

### 📄 **PDF Policy Processing**
- **Automatic text extraction** from PDF policy files
- **Intelligent categorization** based on filenames and content
- **Metadata extraction** (policy type, year, categories)
- **Batch processing** of entire policy directories

### 🤖 **Free LLM Integration**
- **Groq API** - Fast inference with Llama3.1-8B or Mixtral-8x7B
- **Hugging Face** - Local models like DialoGPT-medium
- **Ollama** - Local models (fallback option)
- **Automatic provider selection** and fallback

### 🧠 **Agentic RAG Architecture**
- **Research Agent** - Finds relevant policy documents
- **Analysis Agent** - Extracts key information and requirements
- **Synthesis Agent** - Combines information from multiple sources
- **Communication Agent** - Provides clear, professional responses

### 📊 **Smart Policy Management**
- **10+ Policy Categories**: Travel, Housing, Salary, Medical, Work Conditions, etc.
- **Vector Search** with ChromaDB
- **Similarity-based retrieval**
- **Category filtering**

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Policies  │    │   Image Files   │    │   User Queries  │
│   (Policy file/)│    │   (data/images) │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │PDF Processor│  │OCR Processor│  │Text Cleaner │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Store (ChromaDB)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Embeddings   │  │Metadata     │  │Search Index │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Research     │  │Analysis     │  │Synthesis    │            │
│  │Agent        │  │Agent        │  │Agent        │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐                                              │
│  │Communication│                                              │
│  │Agent        │                                              │
│  └─────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Groq API     │  │Hugging Face │  │Ollama       │            │
│  │(Free Tier)  │  │(Local)      │  │(Local)      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API & Frontend                              │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │FastAPI      │  │Streamlit    │                              │
│  │Backend      │  │Frontend     │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- **Python 3.9+**
- **8GB+ RAM** (for local models)
- **2GB+ free disk space**
- **Internet connection** (for Groq API)

### Optional Dependencies
- **Tesseract OCR** (for image processing)
- **Ollama** (for local LLM models)
- **CUDA** (for GPU acceleration with Hugging Face models)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sopassist-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables (Optional)
```bash
# For Groq API (free tier)
export GROQ_API_KEY="your_groq_api_key"

# For custom LLM provider
export LLM_PROVIDER="groq"  # or "huggingface" or "ollama"
```

### 4. Add Your Policy Files
Place your PDF policy files in the `Policy file/` directory:
```
Policy file/
├── TA-DA Policy (KFL & KML) Dec, 2022.pdf
├── House Rent, Furniture And Cook Allowance Policy.pdf
├── Medical_Bill_Policy.pdf
├── Holiday 2019 (Head Office & Region).pdf
└── ...
```

## 🚀 Quick Start

### Option 1: Interactive Setup
```bash
python start_enhanced_sopassist.py --interactive
```

### Option 2: Command Line Setup
```bash
# Check dependencies
python start_enhanced_sopassist.py --check-deps

# Setup directories
python start_enhanced_sopassist.py --setup

# Process PDF policies
python start_enhanced_sopassist.py --process-pdfs

# Start the complete system
python start_enhanced_sopassist.py --start-all
```

### Option 3: Manual Setup
```bash
# 1. Process PDF policies
python scripts/process_policy_pdfs.py

# 2. Start API server
python api/enhanced_main.py

# 3. Start frontend (in another terminal)
streamlit run frontend/app.py
```

## 📖 Usage

### Web Interface
1. **Start the system**: `python start_enhanced_sopassist.py --start-all`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Ask questions** about your policies

### API Usage
```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/ask", json={
    "question": "What is the vacation policy for employees?",
    "llm_provider": "groq"
})

print(response.json()["answer"])

# Search policies
response = requests.post("http://localhost:8000/search-policies", json={
    "query": "travel allowance",
    "category_filter": "travel"
})

print(response.json())
```

### Direct Script Usage
```bash
# Process specific PDF
python scripts/process_policy_pdfs.py --single-pdf "Policy file/TA-DA Policy.pdf"

# List all processed policies
python scripts/process_policy_pdfs.py --list

# Test search functionality
python scripts/process_policy_pdfs.py --test-search
```

## 🎯 Policy Categories

The system automatically categorizes policies into:

| Category | Keywords | Example Policies |
|----------|----------|------------------|
| **Travel** | ta, da, travel, air, ticket, driver | TA-DA Policy, Air Tickets |
| **Housing** | house, rent, furniture, cook | House Rent Policy |
| **Salary** | salary, increment, bonus, compensation | Salary Increment Proposal |
| **Medical** | medical, health, bill, treatment | Medical Bill Policy |
| **Work Conditions** | holiday, overtime, night, shift | Holiday Policy, Overtime |
| **Allowances** | allowance, hardship, location, uniform | Hardship Allowance |
| **Employee Management** | recruitment, retirement, termination | Retirement Policy |
| **Financial** | financial, budget, expense, claim | Financial Limits |
| **Farm Operations** | farm, hatchery, production, bio | Farm Management Policy |
| **General** | policy, circular, notice, order | General Policies |

## 🤖 LLM Providers

### 1. **Groq API** (Recommended)
- **Free tier**: 100 requests/minute
- **Models**: Llama3.1-8B, Mixtral-8x7B
- **Speed**: Very fast (sub-second responses)
- **Setup**: Requires API key

### 2. **Hugging Face** (Local)
- **Models**: DialoGPT-medium, Falcon-7B
- **Cost**: Free (local processing)
- **Speed**: Moderate (depends on hardware)
- **Setup**: Automatic model download

### 3. **Ollama** (Local)
- **Models**: Llama3, Mistral, CodeLlama
- **Cost**: Free (local processing)
- **Speed**: Fast (with good hardware)
- **Setup**: Requires Ollama installation

## 🔧 Configuration

### Environment Variables
```bash
# LLM Configuration
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq  # groq, huggingface, ollama

# API Configuration
API_PORT=8000
STREAMLIT_PORT=8501

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

### Settings File (`config/settings.py`)
```python
# Free LLM Settings
FREE_LLM_CONFIG = {
    "groq": {
        "default_model": "llama3-8b-8192",
        "timeout": 30,
        "max_tokens": 1000
    },
    "huggingface": {
        "default_model": "microsoft/DialoGPT-medium",
        "device": "auto",
        "max_length": 512
    }
}

# Policy Categories
POLICY_CATEGORIES = {
    "travel": ["ta", "da", "travel", "air", "ticket"],
    "housing": ["house", "rent", "furniture", "cook"],
    # ... more categories
}
```

## 📊 API Endpoints

### Core Endpoints
- `POST /ask` - Ask questions about policies
- `POST /analyze-policy` - Analyze specific policy documents
- `POST /get-recommendations` - Get policy recommendations
- `POST /search-policies` - Search for relevant policies
- `POST /upload-pdf` - Upload and process new PDF policies

### Management Endpoints
- `GET /stats` - Get system statistics
- `GET /health` - Health check
- `DELETE /documents/{id}` - Delete documents

### Example API Usage
```bash
# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?", "llm_provider": "groq"}'

# Search policies
curl -X POST "http://localhost:8000/search-policies" \
  -H "Content-Type: application/json" \
  -d '{"query": "travel allowance", "top_k": 5}'

# Upload PDF
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@policy.pdf"
```

## 🧪 Testing

### Run Basic Tests
```bash
python start_enhanced_sopassist.py --test
```

### Test Specific Components
```bash
# Test PDF processing
python core/pdf_processor.py

# Test free LLM interface
python core/free_llm_interface.py

# Test enhanced agents
python agents/enhanced_policy_agents.py

# Test API
python -m pytest tests/
```

## 📁 Project Structure

```
sopassist-ai/
├── 📄 Policy file/              # Your PDF policy files
├── 🏗️ api/
│   ├── main.py                  # Original API
│   └── enhanced_main.py         # Enhanced API with PDF support
├── 🎨 frontend/
│   └── app.py                   # Streamlit frontend
├── 🔧 core/
│   ├── pdf_processor.py         # PDF text extraction
│   ├── free_llm_interface.py    # Free LLM integration
│   ├── vector_store.py          # ChromaDB integration
│   ├── ocr_processor.py         # Image OCR processing
│   └── llm_interface.py         # Ollama integration
├── 🤖 agents/
│   ├── policy_agents.py         # Original agents
│   └── enhanced_policy_agents.py # Enhanced agentic RAG
├── 📜 scripts/
│   ├── process_policy_pdfs.py   # PDF processing script
│   └── process_images.py        # Image processing script
├── ⚙️ config/
│   └── settings.py              # Configuration settings
├── 🧪 tests/
│   └── test_basic.py            # Basic tests
├── 📊 data/
│   ├── processed_pdfs/          # Extracted text files
│   ├── embeddings/              # Vector store data
│   └── images/                  # Image files
├── 📝 requirements.txt          # Python dependencies
├── 🚀 start_enhanced_sopassist.py # Enhanced startup script
└── 📖 ENHANCED_README.md        # This file
```

## 🔍 Troubleshooting

### Common Issues

#### 1. **PDF Processing Fails**
```bash
# Check if PDFs are readable
python core/pdf_processor.py

# Force reprocess
python start_enhanced_sopassist.py --process-pdfs --force-reprocess
```

#### 2. **LLM Provider Not Working**
```bash
# Check available providers
python start_enhanced_sopassist.py --info

# Test specific provider
python core/free_llm_interface.py
```

#### 3. **Vector Store Issues**
```bash
# Check vector store status
python scripts/process_policy_pdfs.py --list

# Rebuild vector store
rm -rf data/embeddings/
python start_enhanced_sopassist.py --process-pdfs
```

#### 4. **Memory Issues**
```bash
# Use smaller models
export LLM_PROVIDER="groq"  # Use cloud API instead of local

# Reduce batch size in config/settings.py
BATCH_SIZE = 16  # Instead of 32
```

### Performance Optimization

#### For Large Policy Collections
1. **Use Groq API** for faster responses
2. **Increase chunk size** in PDF processing
3. **Use GPU** for Hugging Face models
4. **Optimize vector search** parameters

#### For Limited Resources
1. **Use smaller models** (DialoGPT-medium)
2. **Reduce batch sizes**
3. **Process PDFs in smaller batches**
4. **Use CPU-only mode**

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast, free LLM API
- **Hugging Face** for open-source models
- **CrewAI** for agent orchestration
- **ChromaDB** for vector storage
- **FastAPI** and **Streamlit** for web interfaces

## 📞 Support

For support and questions:
- **Issues**: Create an issue on GitHub
- **Documentation**: Check this README and inline code comments
- **Community**: Join our discussions

---

**🎯 Ready to transform your policy management with AI? Start with `python start_enhanced_sopassist.py --interactive`!** 