# How the RAG System Works: Complete Processing Flow

## 📋 Overview

This document explains how the Enhanced RAG Chatbot processes user queries, uploaded images, and uploaded documents to generate responses.

---

## 🔄 Complete Query Processing Flow

### **Step-by-Step: From User Query to Response**

```
┌─────────────────────────────────────────────────────────────┐
│  USER ENTERS QUERY                                           │
│  Example: "What are the departure times for Bus 91M?"      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Query Preprocessing                                │
│  Location: enhanced_rag_chatbot.py (lines 807-822)         │
│                                                             │
│  Checks:                                                    │
│  • Is this an image query? (get_uploaded_image_filename)   │
│  • Is this self-contained? (is_self_contained_query)        │
│  • Is this factual/real-time? (is_factual_query)            │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌──────────────────┴──────────────────┐
        │                                       │
        ↓                                       ↓
┌───────────────────┐              ┌───────────────────┐
│  IMAGE QUERY?     │              │  TEXT QUERY        │
│  (Direct Vision)  │              │  (RAG Processing)  │
└───────────────────┘              └───────────────────┘
        │                                       │
        │                                       ↓
        │              ┌─────────────────────────────────────┐
        │              │  STEP 2: Query Classification      │
        │              │  Location: query_classifier.py      │
        │              │                                     │
        │              │  Uses GPT-4o to classify:           │
        │              │  • local_rag: Uploaded docs         │
        │              │  • web_search: Real-time data     │
        │              │  • hybrid: Both needed             │
        │              └─────────────────────────────────────┘
        │                                       │
        │                                       ↓
        │              ┌─────────────────────────────────────┐
        │              │  STEP 3: Source Routing             │
        │              │  Location: intelligent_source_router │
        │              │                                     │
        │              │  Routes to appropriate retrieval:   │
        │              │  • Local RAG (FAISS)                │
        │              │  • Web Search (OpenAI/Tavily)       │
        │              │  • Hybrid (Both)                   │
        │              └─────────────────────────────────────┘
        │                                       │
        │                                       ↓
        │              ┌─────────────────────────────────────┐
        │              │  STEP 4: Retrieval & Reranking       │
        │              │                                     │
        │              │  Local RAG:                         │
        │              │  • FAISS vector search (top 10)      │
        │              │  • Cross-Encoder reranking (top 4)   │
        │              │                                     │
        │              │  Web Search:                        │
        │              │  • OpenAI Web Search (real-time)    │
        │              │  • Tavily Search (general)           │
        │              │  • Combine & deduplicate            │
        │              └─────────────────────────────────────┘
        │                                       │
        │                                       ↓
        │              ┌─────────────────────────────────────┐
        │              │  STEP 5: Context Assembly            │
        │              │                                     │
        │              │  • Format retrieved documents       │
        │              │  • Add conversation history (last 6) │
        │              │  • Prepare for LLM                 │
        │              └─────────────────────────────────────┘
        │                                       │
        │                                       ↓
        │              ┌─────────────────────────────────────┐
        │              │  STEP 6: LLM Response Generation    │
        │              │  Location: generate_response_with_  │
        │              │            context()                │
        │              │                                     │
        │              │  • Model: GPT-4o                    │
        │              │  • System prompt based on datasource  │
        │              │  • Includes context + history        │
        │              └─────────────────────────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Response Display                                   │
│  • Show answer                                              │
│  • Display sources (📚 X Sources Used)                      │
│  • Show performance metrics (⏱️ Performance Metrics)        │
│  • Show routing details (🔍 Source Information)               │
│  • Save to chat history                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📄 Document Processing Flow

### **How Uploaded Documents Are Processed**

When you upload a PDF or image file, here's what happens:

#### **1. File Upload & Detection**

- **Location:** `enhanced_rag_chatbot.py` lines 709-720
- User uploads file(s) via Streamlit file uploader
- System checks file type (PDF, PNG, JPG, etc.)
- Skips already processed files (prevents duplicates)

#### **2. Text Extraction**

**For PDF Files:**

- **Location:** `enhanced_rag_chatbot.py` lines 57-70
- **Method:** PyPDF2 library
- **Process:**
  1. Reads PDF file page by page
  2. Extracts text from each page
  3. Adds page number markers: `[Page 1]`, `[Page 2]`, etc.
  4. Combines all pages into single text string
  5. Handles extraction errors gracefully

**For Image Files:**

- **Location:** `enhanced_rag_chatbot.py` lines 73-181
- **Dual Processing Strategy:**

  **A. EasyOCR Text Extraction** (lines 92-101)

  - Extracts visible text from images
  - Handles text-heavy images (documents, signs, labels)
  - Fast processing

  **B. GPT Vision API Analysis** (lines 103-155)

  - Visual understanding and context
  - Describes objects, scenes, locations, landmarks
  - Identifies universities, sculptures, buildings
  - Explains symbolic meanings
  - Provides comprehensive image description

  **C. Intelligent Combination** (lines 157-176)

  - If both OCR and Vision succeed: Combines both with section markers
  - If only OCR succeeds: Uses OCR text only
  - If only Vision succeeds: Uses Vision description only
  - Result: Comprehensive representation for search

#### **3. Text Chunking**

- **Location:** `enhanced_rag_chatbot.py` lines 227-244
- **Method:** RecursiveCharacterTextSplitter (LangChain)
- **Parameters:**
  - Chunk size: 1000 characters
  - Chunk overlap: 200 characters
- **Why:**
  - 1000 chars: Good balance between context and granularity
  - 200 overlap: Prevents information loss at boundaries
  - Recursive splitting: Handles various text structures

#### **4. Embedding Creation**

- **Location:** `enhanced_rag_chatbot.py` lines 246-252
- **Model:** OpenAI Embeddings (`text-embedding-ada-002` or similar)
- **Process:**
  1. Each text chunk is converted to a vector (1536 dimensions)
  2. Embeddings capture semantic meaning
  3. Similar content has similar vectors

#### **5. Vector Store Indexing**

- **Location:** `enhanced_rag_chatbot.py` lines 254-272
- **Store:** FAISS (Facebook AI Similarity Search)
- **Process:**
  1. Creates FAISS index from embeddings
  2. Stores vectors with metadata (source file, chunk index)
  3. Enables fast similarity search
  4. Supports incremental updates (can add new documents)

#### **6. Storage**

- **Location:** `enhanced_rag_chatbot.py` lines 259-263
- Vector store saved in `st.session_state.vectorstore`
- Image files stored in `st.session_state.uploaded_image_files`
- Processed file list in `st.session_state.uploaded_files_processed`

**Complete Document Processing Pipeline:**

```
PDF/Image Upload
      ↓
Text Extraction (PyPDF2 / EasyOCR + GPT Vision)
      ↓
Text Chunking (1000 chars, 200 overlap)
      ↓
Embedding Creation (OpenAI Embeddings)
      ↓
FAISS Vector Store Indexing
      ↓
Ready for Retrieval
```

---

## 🖼️ Image Query Processing Flow

### **How Questions About Uploaded Images Are Handled**

#### **Method 1: Direct GPT Vision API (Preferred)**

**When Used:**

- Query is detected as image-related
- Image file is available in session state
- **Location:** `enhanced_rag_chatbot.py` lines 382-433

**Process:**

1. **Image Detection** (lines 436-491)

   - Checks if query mentions image/picture/photo
   - Detects content queries: "identify", "what is this", "describe"
   - If only one image uploaded, automatically uses it
   - If multiple images, matches by filename

2. **Direct Vision API Call** (lines 405-425)

   - Retrieves image bytes from session state
   - Converts to base64 format
   - Sends image + question directly to GPT-4o Vision API
   - Gets real-time analysis and answer
   - **No RAG processing needed** - direct answer

3. **Response Display**
   - Shows answer immediately
   - Displays image source
   - Shows performance metrics

**Example Flow:**

```
User: "What is this sculpture?"
      ↓
System detects image query
      ↓
Retrieves image from session state
      ↓
Sends to GPT Vision API with question
      ↓
Returns: "This is the Alma Mater sculpture at HKUST..."
```

#### **Method 2: RAG-Based Image Retrieval (Fallback)**

**When Used:**

- Image was processed and indexed in vector store
- Query doesn't trigger direct vision detection
- **Location:** Standard RAG flow (lines 856-950)

**Process:**

1. Image was previously processed:

   - EasyOCR extracted text
   - GPT Vision generated description
   - Combined text was chunked and indexed

2. Query processed through normal RAG:

   - Query classified (likely `local_rag`)
   - FAISS retrieves relevant chunks
   - Reranking applied
   - Context sent to LLM

3. LLM generates answer based on indexed image description

**Comparison:**

- **Direct Vision:** More accurate, real-time analysis
- **RAG-Based:** Uses stored description, may miss details

---

## 🔍 Query Classification & Routing

### **How the System Decides Where to Search**

**Location:** `query_classifier.py`

#### **Classification Process:**

1. **LLM-Based Classification** (Primary)

   - Uses GPT-4o to analyze query
   - Considers:
     - Query content and intent
     - Whether documents are uploaded
     - Real-time vs. document-specific needs
   - Returns: `local_rag`, `web_search`, or `hybrid`

2. **Fallback Classification** (If LLM fails)

   - Keyword-based detection
   - Checks for:
     - Document keywords: "my document", "uploaded", "pdf"
     - Real-time keywords: "latest", "current", "today", "stock", "bus"
     - Specific content: "sculpture", "building", "image"

3. **Smart Override Logic**
   - If documents uploaded + specific query → prefer `local_rag`
   - If real-time query → force `web_search`
   - If no documents → always `web_search`

#### **Routing Decision Examples:**

| Query                        | Documents? | Classification | Reasoning                        |
| ---------------------------- | ---------- | -------------- | -------------------------------- |
| "What is this sculpture?"    | Yes        | `local_rag`    | Specific content query with docs |
| "Bus 91M departure times"    | No         | `web_search`   | Real-time data needed            |
| "Latest AI news"             | Yes        | `web_search`   | Real-time query overrides docs   |
| "Compare my doc with trends" | Yes        | `hybrid`       | Needs both local + web           |

---

## 🔎 Retrieval Process

### **Local RAG Retrieval**

**Location:** `intelligent_source_router.py` lines 472-551

**Process:**

1. **FAISS Vector Search**

   - Embeds user query using same embedding model
   - Performs cosine similarity search in FAISS
   - Retrieves top 10 most similar document chunks

2. **Reranking** (if multiple docs found)

   - **Primary:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
     - Processes query-document pairs together
     - Computes relevance scores
     - Sorts by score
   - **Fallback:** BM25 (if Cross-Encoder fails)
     - Statistical keyword-based ranking
     - Fast and lightweight
   - Returns top 4 most relevant documents

3. **Context Formatting**
   - Formats documents with metadata (source, page, chunk)
   - Prepares for LLM consumption

### **Web Search Retrieval**

**Location:** `intelligent_source_router.py` lines 335-470

**Process:**

1. **Real-Time Query Detection**

   - Checks for keywords: "departure", "stock", "weather", "schedule"
   - If detected, prioritizes OpenAI Web Search

2. **Multi-Source Search**

   - **OpenAI Web Search** (if real-time query)
   - **Tavily Search** (always, for general queries)
   - **Fallback Chain:** Wikipedia → ArXiv → Google → Bing → Mock

3. **Result Merging**

   - Combines results from OpenAI + Tavily
   - Removes duplicates by URL
   - Limits to max_results (default: 5)

4. **Targeted Searches** (if needed)
   - Detects multi-part queries: "Who won X and what is Y?"
   - Detects specific info requests: "Douban score", "IMDB rating"
   - Performs additional searches for missing information

---

## 💬 Response Generation

### **How the Final Answer is Generated**

**Location:** `enhanced_rag_chatbot.py` lines 494-698

#### **Process:**

1. **Context Assembly**

   - Combines retrieved documents/web results
   - Adds conversation history (last 6 messages)
   - Formats for LLM

2. **System Prompt Selection**

   - Based on datasource (`local_rag`, `web_search`, `hybrid`)
   - Includes:
     - Current date (for date-aware queries)
     - Conversation history context
     - Date-aware instructions (for "next"/"earliest" queries)
     - Specific instructions for extracting data from search results

3. **LLM Generation**

   - **Model:** GPT-4o
   - **Temperature:** 0.5 (balanced creativity/consistency)
   - Processes query with context
   - Generates answer

4. **Response Enhancement**
   - Extracts specific data from search results
   - Handles multi-part queries
   - Compares dates for "next" queries
   - Uses conversation history for follow-ups

#### **Special Handling:**

**Date-Aware Queries:**

- Detects: "next", "earliest", "upcoming", "first"
- Instructions: Compare ALL dates in results
- Find closest date to current date

**Multi-Part Queries:**

- Detects: "Who won X and what is Y?"
- Handles each part separately
- Indicates which parts found/missing

**Follow-Up Queries:**

- Uses conversation history
- Understands vague references: "show me all", "what about that"
- Maintains context across messages

---

## ⏱️ Performance Tracking

### **What Gets Measured**

**Location:** `enhanced_rag_chatbot.py` lines 857-912

**Metrics Tracked:**

1. **Search/Routing Time**

   - Query classification
   - Retrieval (FAISS/web search)
   - Reranking
   - Context assembly

2. **LLM Generation Time**

   - Time to generate response
   - From LLM call start to completion

3. **Total Response Time**
   - Sum of search + LLM time
   - End-to-end query processing

**Display:**

- Always shown in expandable "⏱️ Performance Metrics" section
- Persisted in chat history
- Available for every assistant message

---

## 🎯 Key Design Decisions

### **Why These Choices?**

1. **Dual Image Processing (OCR + Vision)**

   - OCR: Fast text extraction
   - Vision: Contextual understanding
   - Combined: Comprehensive representation

2. **Two-Stage Retrieval (FAISS + Reranking)**

   - FAISS: Fast initial search (top 10)
   - Reranking: Accurate relevance (top 4)
   - Balance: Speed + Accuracy

3. **Multi-Source Web Search**

   - OpenAI: Real-time data
   - Tavily: General queries
   - Fallbacks: Reliability
   - Combined: Comprehensive coverage

4. **Conversation Memory (Last 6 Messages)**

   - Enables follow-up questions
   - Maintains context
   - Handles vague references

5. **Direct Vision for Image Queries**
   - More accurate than stored descriptions
   - Real-time analysis
   - Better for specific questions

---

## 📊 Summary: Complete Processing Flow

### **For Text Queries:**

```
Query → Classification → Routing → Retrieval → Reranking → Context Assembly → LLM → Response
```

### **For Image Queries:**

```
Query → Image Detection → Direct GPT Vision → Response
```

### **For Document Upload:**

```
Upload → Extraction → Chunking → Embedding → FAISS Indexing → Ready for Retrieval
```

### **For Image Upload:**

```
Upload → OCR + Vision → Combined Text → Chunking → Embedding → FAISS Indexing → Ready for Retrieval
```

---

## 🔗 Integration Points

All components work together:

1. **QueryClassifier** analyzes intent → routes to source
2. **IntelligentSourceRouter** executes retrieval → applies reranking
3. **WebSearchEnhanced** searches multiple APIs → combines results
4. **Local RAG** retrieves from FAISS → reranks → formats context
5. **Response Generator** combines context + history → generates answer

**Result:** A production-ready RAG system that handles diverse queries with high accuracy and reliability.
