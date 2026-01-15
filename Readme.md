# 🚀 NLP RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that supports multiple file types and advanced AI models.

## ✨ Key Features

- **Multi-Modal Support**: PDF documents and images (PNG, JPG, JPEG, GIF, BMP)
- **Auto-Processing**: Files are processed immediately upon upload
- **OpenAI GPT-4o**: Latest and most efficient model for text generation
- **Conversational Memory**: AI remembers previous conversations
- **Source Attribution**: AI references specific files when providing information
- **Session Management**: Multiple chat sessions with easy switching

## 🛠️ Technical Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS
- **Framework**: Streamlit + LangChain
- **OCR**: EasyOCR for image text extraction

## 📋 Quick Start

1. **Install dependencies**(python 3.9)

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**

   ```bash
   echo "OPENAI_API_KEY=yourAPIkey" > .env
   ```

   add TAVILY_API_KEY to .env

3. **Run the application**
   ```bash
   streamlit run Home_Page.py
   ```

## 🚀 Usage

1. **Upload Files**: Drag and drop PDFs or images in the sidebar
2. **Auto-Processing**: Files are processed automatically
3. **Start Chatting**: Ask questions about your documents
4. **File Queries**: Ask "What files did I upload?" to see processed documents

## 🎯 Key Improvements

| Feature        | Original              | Enhanced                  |
| -------------- | --------------------- | ------------------------- |
| **File Types** | PDF only              | PDF + Images              |
| **AI Model**   | Gemini Pro            | **GPT-4o**                |
| **Embeddings** | Google (quota issues) | OpenAI (no limits)        |
| **Memory**     | Basic                 | Full conversation history |
| **Processing** | Manual                | Auto-processing           |

## 📁 Project Structure

```
├── pages/
│   ├── 🚀_NLP_RAG_Chatbot.py    # Main RAG chatbot
│   ├── 📄_PDF_Chatbot.py         # PDF-only chatbot
│   ├── 🖼️_Image_Chatbot.py       # Image chatbot
│   └── 💬_Narrative_Chatbot.py   # Narrative chatbot
├── data/                         # Chat history storage
├── faiss_index/                  # Vector store files
└── requirements.txt             # Dependencies
```

## 🔧 Configuration

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Model Settings:**

- LLM: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`
- Temperature: 0.1
- Max Tokens: 2048

---

**Built for NLP course RAG project with advanced document processing capabilities**
# NLP_Project
