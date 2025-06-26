# 🎯 Enhanced SopAssist AI - Agentic RAG Policy Assistant

A comprehensive AI-powered policy assistant that processes PDF policy documents and provides intelligent answers using **free LLM models** (Groq, Hugging Face) and **agentic RAG** with CrewAI. **Now with full Bengali language support!** 🇧🇩

## 🚀 Key Features

### 📄 **PDF Policy Processing**
- **Automatic text extraction** from PDF policy files
- **Intelligent categorization** based on filenames and content
- **Metadata extraction** (policy type, year, categories)
- **Batch processing** of entire policy directories
- **🌍 Bengali language support** with OCR

### 🤖 **Free LLM Integration**
- **Groq API** - Fast inference with Llama3.1-8B or Mixtral-8x7B
- **Hugging Face** - Local models like DialoGPT-medium
- **Ollama** - Local models (fallback option)
- **Automatic provider selection** and fallback
- **🌍 Bilingual responses** (English & Bengali)

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
- **🌍 Bengali keyword recognition**

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Policies  │    │   Image Files   │    │   User Queries  │
│   (Policy file/)│    │   (data/images) │    │                 │
│   🌍 Bengali    │    │   🌍 Bengali    │    │   🌍 Bengali    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │PDF Processor│  │OCR Processor│  │Text Cleaner │            │
│  │🌍 Bengali   │  │🌍 Bengali   │  │🌍 Bengali   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Store (ChromaDB)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Embeddings   │  │Metadata     │  │Search Index │            │
│  │🌍 Bengali   │  │🌍 Language  │  │🌍 Bengali   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Research     │  │Analysis     │  │Synthesis    │            │
│  │Agent        │  │Agent        │  │Agent        │            │
│  │🌍 Bengali   │  │🌍 Bengali   │  │🌍 Bengali   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐                                              │
│  │Communication│                                              │
│  │Agent        │                                              │
│  │🌍 Bengali   │                                              │
│  └─────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Groq API     │  │Hugging Face │  │Ollama       │            │
│  │(Free Tier)  │  │(Local)      │  │(Local)      │            │
│  │🌍 Bengali   │  │🌍 Bengali   │  │🌍 Bengali   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API & Frontend                              │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │FastAPI      │  │Streamlit    │                              │
│  │Backend      │  │Frontend     │                              │
│  │🌍 Bengali   │  │🌍 Bengali   │                              │
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
- **Bengali language pack** for Tesseract
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

### 3. Setup Bengali Language Support
```bash
# Automatic setup
python scripts/setup_bengali_support.py

# Or manual setup
python scripts/setup_bengali_support.py --manual
```

### 4. Setup Environment Variables (Optional)
```bash
# For Groq API (free tier)
export GROQ_API_KEY="your_groq_api_key"

# For custom LLM provider
export LLM_PROVIDER="groq"  # or "huggingface" or "ollama"
```

### 5. Add Your Policy Files
Place your PDF policy files in the `Policy file/` directory:
```
Policy file/
├── TA-DA Policy (KFL & KML) Dec, 2022.pdf
├── House Rent, Furniture And Cook Allowance Policy.pdf
├── Medical_Bill_Policy.pdf
├── Holiday 2019 (Head Office & Region).pdf
├── বাংলা নীতি নথি.pdf  # Bengali policy documents
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

# Process PDF policies (including Bengali)
python start_enhanced_sopassist.py --process-pdfs

# Start the complete system
python start_enhanced_sopassist.py --start-all
```

### Option 3: Manual Setup
```bash
# 1. Setup Bengali support
python scripts/setup_bengali_support.py

# 2. Process PDF policies
python scripts/process_policy_pdfs.py

# 3. Start API server
python api/enhanced_main.py

# 4. Start frontend (in another terminal)
streamlit run frontend/app.py
```

## 📖 Usage

### Web Interface
1. **Start the system**: `python start_enhanced_sopassist.py --start-all`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Ask questions** about your policies in English or Bengali

### API Usage
```python
import requests

# Ask a question in English
response = requests.post("http://localhost:8000/ask", json={
    "question": "What is the vacation policy for employees?",
    "llm_provider": "groq"
})

print(response.json()["answer"])

# Ask a question in Bengali
response = requests.post("http://localhost:8000/ask", json={
    "question": "কর্মচারীদের ছুটির নীতি কী?",
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

## 🎯 Policy Categories (English + Bengali)

The system automatically categorizes policies into:

| Category | English Keywords | Bengali Keywords | Example Policies |
|----------|------------------|------------------|------------------|
| **Travel** | ta, da, travel, air, ticket, driver | ভ্রমণ, টিকিট, ভ্রমণ ভাতা | TA-DA Policy, Air Tickets |
| **Housing** | house, rent, furniture, cook | বাড়ি, ভাড়া, ফার্নিচার, রান্না | House Rent Policy |
| **Salary** | salary, increment, bonus, compensation | বেতন, বোনাস, বৃদ্ধি, মজুরি | Salary Increment Proposal |
| **Medical** | medical, health, bill, treatment | চিকিৎসা, স্বাস্থ্য, বিল, চিকিৎসা বিল | Medical Bill Policy |
| **Work Conditions** | holiday, overtime, night, shift | ছুটি, অতিরিক্ত, রাত, শিফট | Holiday Policy, Overtime |
| **Allowances** | allowance, hardship, location, uniform | ভাতা, কষ্ট, ইউনিফর্ম, মোবাইল | Hardship Allowance |
| **Employee Management** | recruitment, retirement, termination | নিয়োগ, অবসর, বরখাস্ত, নোটিশ | Retirement Policy |
| **Financial** | financial, budget, expense, claim | আর্থিক, বাজেট, খরচ, দাবি | Financial Limits |
| **Farm Operations** | farm, hatchery, production, bio | খামার, হ্যাচারি, উৎপাদন, জৈব | Farm Management Policy |
| **General** | policy, circular, notice, order | নীতি, পরিপত্র, নোটিশ, আদেশ | General Policies |

## 🌍 Bengali Language Support

### Features
- **Automatic language detection** in PDFs and queries
- **Bengali OCR** with Tesseract
- **Bilingual categorization** with Bengali keywords
- **Bengali responses** from LLM models
- **Bengali text processing** and cleaning

### Setup Bengali Support
```bash
# Check current installation
python scripts/setup_bengali_support.py --check

# Install Bengali language pack
python scripts/setup_bengali_support.py --install

# Test Bengali OCR
python scripts/setup_bengali_support.py --test

# Show manual instructions
python scripts/setup_bengali_support.py --manual
```

### Manual Installation
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-ben

# CentOS/RHEL/Fedora
sudo yum install tesseract tesseract-langpack-ben
# or
sudo dnf install tesseract tesseract-langpack-ben

# macOS
brew install tesseract tesseract-lang

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install with Bengali language pack
```

## 🤖 LLM Providers

### 1. **Groq API** (Recommended)
- **Free tier**: 100 requests/minute
- **Models**: Llama3.1-8B, Mixtral-8x7B
- **Speed**: Very fast (sub-second responses)
- **Setup**: Requires API key
- **🌍 Bengali**: Excellent support

### 2. **Hugging Face** (Local)
- **Models**: DialoGPT-medium, Falcon-7B
- **Cost**: Free (local processing)
- **Speed**: Moderate (depends on hardware)
- **Setup**: Automatic model download
- **🌍 Bengali**: Good support

### 3. **Ollama** (Local)
- **Models**: Llama3, Mistral, CodeLlama
- **Cost**: Free (local processing)
- **Speed**: Fast (with good hardware)
- **Setup**: Requires Ollama installation
- **🌍 Bengali**: Limited support

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
# Language Support Settings
LANGUAGE_CONFIG = {
    "supported_languages": ["english", "bengali"],
    "default_language": "english",
    "bengali_detection_threshold": 0.1,
    "bengali_ocr_lang": "ben+eng",
    "english_ocr_lang": "eng"
}

# Policy Categories (English + Bengali)
POLICY_CATEGORIES = {
    "travel": ["ta", "da", "travel", "air", "ticket", "driver", "trip", "allowance", "ভ্রমণ", "টিকিট", "ভ্রমণ ভাতা"],
    "housing": ["house", "rent", "furniture", "cook", "accommodation", "বাড়ি", "ভাড়া", "ফার্নিচার", "রান্না"],
    # ... more categories
}
```

## 📊 API Endpoints

### Core Endpoints
- `POST /ask` - Ask questions about policies (English & Bengali)
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
# Ask a question in English
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the vacation policy?", "llm_provider": "groq"}'

# Ask a question in Bengali
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "ছুটির নীতি কী?", "llm_provider": "groq"}'

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

### Test Bengali Support
```bash
# Test Bengali OCR
python scripts/setup_bengali_support.py --test

# Test Bengali PDF processing
python core/pdf_processor.py

# Test Bengali LLM responses
python core/free_llm_interface.py
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
├── 📄 Policy file/              # Your PDF policy files (English & Bengali)
├── 🏗️ api/
│   ├── main.py                  # Original API
│   └── enhanced_main.py         # Enhanced API with PDF & Bengali support
├── 🎨 frontend/
│   └── app.py                   # Streamlit frontend
├── 🔧 core/
│   ├── pdf_processor.py         # PDF text extraction (Bengali support)
│   ├── free_llm_interface.py    # Free LLM integration (Bengali support)
│   ├── vector_store.py          # ChromaDB integration
│   ├── ocr_processor.py         # Image OCR processing
│   └── llm_interface.py         # Ollama integration
├── 🤖 agents/
│   ├── policy_agents.py         # Original agents
│   └── enhanced_policy_agents.py # Enhanced agentic RAG
├── 📜 scripts/
│   ├── process_policy_pdfs.py   # PDF processing script
│   ├── process_images.py        # Image processing script
│   └── setup_bengali_support.py # Bengali language setup
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

#### 1. **Bengali OCR Not Working**
```bash
# Check Bengali language pack
python scripts/setup_bengali_support.py --check

# Install Bengali support
python scripts/setup_bengali_support.py --install

# Test Bengali OCR
python scripts/setup_bengali_support.py --test
```

#### 2. **PDF Processing Fails**
```bash
# Check if PDFs are readable
python core/pdf_processor.py

# Force reprocess
python start_enhanced_sopassist.py --process-pdfs --force-reprocess
```

#### 3. **LLM Provider Not Working**
```bash
# Check available providers
python start_enhanced_sopassist.py --info

# Test specific provider
python core/free_llm_interface.py
```

#### 4. **Vector Store Issues**
```bash
# Check vector store status
python scripts/process_policy_pdfs.py --list

# Rebuild vector store
rm -rf data/embeddings/
python start_enhanced_sopassist.py --process-pdfs
```

#### 5. **Bengali Text Not Recognized**
```bash
# Check language detection
python core/pdf_processor.py

# Verify Bengali characters in text
python -c "print('বাংলা' in 'বাংলা ভাষা')"
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
- **Tesseract** for Bengali OCR support

## 📞 Support

For support and questions:
- **Issues**: Create an issue on GitHub
- **Documentation**: Check this README and inline code comments
- **Community**: Join our discussions

---

**🎯 Ready to transform your policy management with AI? Start with `python start_enhanced_sopassist.py --interactive`!**

**🌍 Now with full Bengali language support for your local policy documents!** 🇧🇩 