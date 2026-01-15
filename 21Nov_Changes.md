# Changes Made - November 11, 2024

## Overview
This document summarizes all changes made to the NLP RAG Chatbot project to improve image handling, fix dependency conflicts, and enhance query routing logic.

---

## 1. Dependency Conflict Resolutions

### File: `requirements.txt`

**Changes:**
- Added dependency conflict resolutions section
- Updated SQLAlchemy constraint from `<2.0.0` to `>=2.0.0` (to fix langchain_community compatibility)
- Added explicit version constraints for conflicting packages

**Added Dependencies:**
```txt
# Dependency conflict resolutions
tqdm>=4.65.0
sqlalchemy>=2.0.0
websockets>=10.0,<13.0
ml-dtypes>=0.5.0
```

**Issues Fixed:**
- ✅ chromadb 0.5.21 requires tqdm>=4.65.0 (was 4.62.3)
- ✅ dataset 1.6.2 requires sqlalchemy<2.0.0 (but langchain_community needs 2.0+)
- ✅ gradio-client 1.4.3 requires websockets<13.0 (was 14.2)
- ✅ onnx 1.19.1 requires ml_dtypes>=0.5.0 (was 0.2.0)
- ✅ ultralytics 8.0.219 requires tqdm>=4.64.0 (was 4.62.3)

**Note:** SQLAlchemy was upgraded to 2.0.44 to support langchain_community's `Executable` import requirement.

---

## 2. GPT Vision API Integration

### File: `pages/enhanced_rag_chatbot.py`

### 2.1 New Imports
```python
import base64
import io
from openai import OpenAI
```

### 2.2 Enhanced Image Extraction Function

**Function: `extract_text_from_image()`**

**Changes:**
- Now uses both EasyOCR and GPT Vision API
- EasyOCR extracts text from images
- GPT Vision API analyzes visual content (objects, scenes, descriptions)
- Enhanced prompt to request:
  - Specific names of landmarks, sculptures, buildings
  - University identification if it's a campus
  - Symbolic meanings of sculptures/monuments
  - Any known information about identifiable objects

**Key Features:**
- Combines OCR text and vision description intelligently
- Handles errors gracefully with fallbacks
- Provides comprehensive image analysis

### 2.3 Direct Image Question Answering

**New Function: `answer_image_question_directly()`**

**Purpose:** Answer questions about uploaded images using GPT Vision API directly (bypassing stored descriptions)

**How it works:**
1. Retrieves image file from session state
2. Converts to base64
3. Sends image + question directly to GPT Vision API (gpt-4o)
4. Returns direct answer about the image

**Benefits:**
- More accurate than using stored descriptions
- Can answer specific questions about images
- Works like ChatGPT's vision feature

### 2.4 Smart Image Query Detection

**New Function: `get_uploaded_image_filename()`**

**Purpose:** Detect if a query is about an uploaded image, even without explicit "image" keywords

**Detection Patterns:**
- Explicit image keywords: 'image', 'picture', 'photo', 'photograph', 'img'
- Content query patterns: 'identify', 'what is this', 'describe', 'explain', 'where is', 'symbolic', 'meaning', 'located'
- Excludes document-specific queries

**Logic:**
- If only one image uploaded → use that image
- If multiple images → try to match by filename
- Detects queries about content in images even without "image" keyword

### 2.5 Session State Updates

**Added:**
```python
if 'uploaded_image_files' not in st.session_state:
    st.session_state.uploaded_image_files = {}  # Store image files: {filename: file_bytes}
```

**Purpose:** Store uploaded image files for direct GPT Vision API access

**Updated File Processing:**
- When processing images, now stores the image file bytes in session state
- Enables direct access to images for GPT Vision queries

### 2.6 Main Chat Loop Updates

**Changes:**
- Added check for image queries before routing
- If image query detected → uses `answer_image_question_directly()`
- Otherwise → uses normal RAG/web search flow
- Fixed duplicate message append bug
- Fixed `routing_result` reference error

---

## 3. Query Routing Improvements

### File: `query_classifier.py`

### 3.1 Updated Router Prompt

**Old Behavior:**
- "ONLY use local_rag if explicitly mentioned documents"
- Defaulted to web_search for most queries

**New Behavior:**
- "PREFER local_rag when documents uploaded"
- Checks uploaded documents first for specific queries
- Only uses web_search for real-time queries

**New Prompt Logic:**
```
1. **local_rag** (PREFERRED when documents uploaded):
   - Questions about specific things that could be in documents
   - Queries about content, descriptions, or information in uploaded files
   - Questions that don't require real-time/current data

2. **web_search** (ONLY for queries that CANNOT be in documents):
   - Current events, news, "latest", "recent", real-time data
   - Weather, stock prices, live updates

3. **hybrid** (When both needed):
   - Compare document with current standards
   - Update document with latest research
```

### 3.2 Smart Override Logic

**Added Logic:**
- Detects queries about specific things (sculptures, buildings, images, objects, people)
- When documents are uploaded and query is about specific content:
  - Overrides `web_search` → `local_rag`
- Only uses `web_search` for real-time queries (news, weather, current events)

**Real-time Keywords:**
```python
['latest', 'current', 'recent', 'today', 'now', 'this week', 'this month',
 'news', 'breaking', 'weather', 'temperature', 'forecast', 'stock', 'price',
 'live', 'happening now', 'update', 'status']
```

**Specific Content Keywords:**
```python
['sculpture', 'building', 'statue', 'monument', 'image', 'picture', 'photo',
 'what is', 'what does', 'describe', 'show', 'about', 'where is', 'what is the']
```

### 3.3 Enhanced Fallback Classification

**Updated Logic:**
1. If no docs uploaded → always web_search
2. If explicit document reference → local_rag
3. If real-time query → web_search
4. If specific content query with docs → local_rag (NEW)
5. If web intent → web_search
6. **Default with docs → local_rag** (CHANGED from web_search)

**Key Change:** Default behavior now prefers local_rag when documents are available.

---

## 4. Performance Optimization

### File: `pages/enhanced_rag_chatbot.py`

**Function: `initialize_source_router()`**

**Change:**
```python
# OLD
fetch_full_content=True

# NEW
fetch_full_content=False  # OPTIMIZED: Disable full content fetching for faster responses
```

**Impact:**
- Faster response times (no full page downloads)
- Uses snippets instead of full content
- Reduces API calls and processing time

---

## 5. Bug Fixes

### 5.1 Fixed `routing_result` Reference Error

**Issue:** `routing_result` was referenced before assignment when image queries were detected.

**Fix:**
- Removed duplicate message append code
- Fixed variable scope in exception handling
- Each code path now properly handles message appending

### 5.2 Fixed SQLAlchemy Import Error

**Issue:** `ImportError: cannot import name 'Executable' from 'sqlalchemy'`

**Root Cause:** 
- langchain_community 0.0.20 requires SQLAlchemy 2.0+ (uses `Executable`)
- But dataset package required SQLAlchemy <2.0.0
- Since dataset isn't used in codebase, upgraded SQLAlchemy to 2.0+

**Fix:**
- Changed SQLAlchemy constraint from `>=1.3.2,<2.0.0` to `>=2.0.0`
- Upgraded to SQLAlchemy 2.0.44

---

## 6. Summary of Key Improvements

### 6.1 Image Handling
- ✅ Can now analyze non-text images (landscapes, sculptures, buildings)
- ✅ Uses GPT Vision API for visual analysis
- ✅ Direct image question answering (bypasses stored descriptions)
- ✅ Smart detection of image queries

### 6.2 Query Routing
- ✅ Prefers checking uploaded documents first
- ✅ Only uses web search for real-time queries
- ✅ Better handling of specific content queries
- ✅ Improved default behavior when documents are available

### 6.3 Performance
- ✅ Faster responses (disabled full content fetching)
- ✅ Optimized routing logic

### 6.4 Stability
- ✅ Fixed all dependency conflicts
- ✅ Fixed SQLAlchemy compatibility issues
- ✅ Fixed variable reference errors

---

## 7. Files Modified

1. **`requirements.txt`**
   - Added dependency conflict resolutions
   - Updated SQLAlchemy version constraint

2. **`pages/enhanced_rag_chatbot.py`**
   - Added GPT Vision API integration
   - Enhanced image extraction
   - Added direct image question answering
   - Added smart image query detection
   - Updated session state management
   - Fixed bugs in chat loop

3. **`query_classifier.py`**
   - Updated router prompt
   - Added smart override logic
   - Enhanced fallback classification
   - Improved default routing behavior

---

## 8. Testing Recommendations

### Test Cases:
1. ✅ Upload a landscape image → Ask "What is the image about?"
2. ✅ Upload a sculpture image → Ask "Identify this sculpture and explain its meaning"
3. ✅ Upload documents → Ask about specific content → Should check documents first
4. ✅ Ask about current events → Should use web search
5. ✅ Upload image + ask specific questions → Should use GPT Vision directly

### Expected Behaviors:
- Image queries should use GPT Vision API directly
- Queries about uploaded content should check documents first
- Real-time queries should use web search
- System should handle errors gracefully

---

## 9. Notes

- **Model Used:** GPT-4o for vision capabilities
- **Image Storage:** Images stored in session state as bytes for direct API access
- **Fallback:** If GPT Vision fails, falls back to stored descriptions
- **Performance:** Disabled full content fetching for faster responses

---

## 10. Future Improvements (Optional)

- [ ] Add support for multiple image queries
- [ ] Cache GPT Vision responses to reduce API calls
- [ ] Add image comparison features
- [ ] Improve error messages for image processing failures
- [ ] Add progress indicators for image processing

---

**Date:** November 11, 2024  
**Author:** AI Assistant  
**Version:** Enhanced RAG Chatbot v2.0

