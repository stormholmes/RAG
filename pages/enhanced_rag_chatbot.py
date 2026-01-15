"""
Enhanced RAG Chatbot with Intelligent Response Generation
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import asyncio
import streamlit as st
from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from PIL import Image
import easyocr
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
import base64
import io
import time
from openai import OpenAI

# Import routing components
from intelligent_source_router import IntelligentSourceRouter
from query_classifier import QueryClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced NLP RAG Chatbot - With Live Web Search",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'source_router' not in st.session_state:
    st.session_state.source_router = None
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = []
if 'uploaded_image_files' not in st.session_state:
    st.session_state.uploaded_image_files = {}  # Store image files: {filename: file_bytes}


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_image(image_file):
    """
    Extract text and analyze image using both EasyOCR and GPT Vision API
    - Uses EasyOCR for text extraction
    - Uses GPT Vision API for visual analysis (objects, scenes, descriptions)
    """
    try:
        # Read image file - reset pointer and load image
        image_file.seek(0)  # Reset file pointer
        image = Image.open(image_file)
        
        # Convert image to base64 for GPT Vision API
        buffered = io.BytesIO()
        # Preserve original format or use PNG
        image_format = image.format if image.format else 'PNG'
        image.save(buffered, format=image_format)
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Step 1: Try EasyOCR for text extraction
        ocr_text = ""
        try:
            # Reset file pointer for EasyOCR
            image_file.seek(0)
            reader = easyocr.Reader(['en'])
            results = reader.readtext(image)
            ocr_text = "\n".join([result[1] for result in results])
            logger.info(f"EasyOCR extracted {len(ocr_text)} characters of text")
        except Exception as e:
            logger.warning(f"EasyOCR error: {e}")
        
        # Step 2: Use GPT Vision API for visual analysis
        vision_description = ""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found, skipping GPT Vision analysis")
            else:
                client = OpenAI(api_key=api_key)
                
                # Prepare the image for vision API
                image_url = f"data:image/png;base64,{image_base64}"
                
                # Create a detailed prompt for image analysis
                vision_prompt = """Analyze this image in detail. Describe:
1. What objects, people, buildings, or scenes are visible
2. The setting, location, or environment
3. Any text visible in the image
4. Colors, composition, and visual elements
5. Any notable features or characteristics
6. If you can identify specific landmarks, sculptures, buildings, or locations, provide their names and any known information about them
7. If this appears to be a university campus, identify which university if possible
8. For sculptures or monuments, describe any symbolic meaning if recognizable

Provide a comprehensive description that would help someone understand what's in this image for search and retrieval purposes. Include any specific names, locations, or contextual information you can identify."""
                
                response = client.chat.completions.create(
                    model="gpt-4o",  # GPT-4o has vision capabilities
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": vision_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                vision_description = response.choices[0].message.content
                logger.info(f"GPT Vision generated {len(vision_description)} characters of description")
                
        except Exception as e:
            logger.warning(f"GPT Vision API error: {e}")
            st.warning(f"Vision analysis error: {e}")
        
        # Step 3: Combine results intelligently
        combined_text = ""
        
        if ocr_text and len(ocr_text.strip()) > 10:
            # If we have substantial text from OCR, include it
            combined_text += f"[TEXT EXTRACTED FROM IMAGE]\n{ocr_text}\n\n"
        
        if vision_description:
            # Always include vision description
            combined_text += f"[IMAGE DESCRIPTION]\n{vision_description}"
        
        # If we have both, add a separator
        if ocr_text and vision_description:
            combined_text = f"[TEXT EXTRACTED FROM IMAGE]\n{ocr_text}\n\n[IMAGE DESCRIPTION]\n{vision_description}"
        elif not combined_text:
            # Fallback if both failed
            combined_text = "[IMAGE] No text or description could be extracted from this image."
            logger.warning("Both OCR and Vision API failed to extract content")
        
        return combined_text
        
    except Exception as e:
        st.error(f"Error extracting image content: {e}")
        logger.error(f"Image extraction error: {e}")
        return ""


def process_uploaded_files(uploaded_files):
    """Process uploaded files and create/update vector store"""
    if not uploaded_files:
        return None

    with st.spinner("Processing uploaded files..."):
        all_texts = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name in st.session_state.uploaded_files_processed:
                continue

            try:
                if file_name.endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                elif file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    text = extract_text_from_image(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {file_name}")
                    continue

                if text:
                    all_texts.append({
                        'content': text,
                        'source': file_name
                    })
                    st.session_state.uploaded_files_processed.append(file_name)
                    
                    # Store image file for direct GPT Vision queries
                    if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        uploaded_file.seek(0)  # Reset file pointer
                        st.session_state.uploaded_image_files[file_name] = uploaded_file.read()
                        logger.info(f"Stored image file: {file_name}")

            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
                logger.error(f"File processing error for {file_name}: {e}")

        if not all_texts:
            return None

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        documents = []
        for item in all_texts:
            chunks = text_splitter.split_text(item['content'])
            for i, chunk in enumerate(chunks):
                documents.append({
                    'page_content': chunk,
                    'metadata': {
                        'source': item['source'],
                        'chunk': i
                    }
                })

        # Create embeddings
        try:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            logger.error(f"Embeddings error: {e}")
            return None

        # Create or update vector store
        texts = [doc['page_content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]

        try:
            if st.session_state.vectorstore is None:
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            else:
                vectorstore = st.session_state.vectorstore
                vectorstore.add_texts(texts, metadatas=metadatas)

            st.success(f"✅ Processed {len(all_texts)} file(s) into {len(documents)} chunks")
            logger.info(f"Processed {len(all_texts)} files into {len(documents)} chunks")
            return vectorstore

        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            logger.error(f"Vector store error: {e}")
            return None


def initialize_source_router():
    """Initialize the intelligent source router"""
    if st.session_state.source_router is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            st.session_state.source_router = IntelligentSourceRouter(
                api_key=api_key,
                enable_web_search=True,
                web_max_results=5,
                fetch_full_content=False  # OPTIMIZED: Disable full content fetching for faster responses
            )
            logger.info("Source router initialized (optimized for speed - snippets only)")
        except Exception as e:
            st.error(f"Error initializing source router: {e}")
            logger.error(f"Source router initialization error: {e}")

def run_async(coroutine):
    """
    Helper to run async coroutines in Streamlit's sync environment.
    Handles existing event loops if necessary.
    """
    try:
        # 尝试获取当前正在运行的事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 如果没有运行中的循环，就创建一个新的
        loop = None

    if loop and loop.is_running():
        # 如果已经有循环在运行（这在某些 Streamlit 部署中很少见，但为了稳健性）
        # 这里使用 nest_asyncio 或者创建任务，但在纯脚本中最简单的是报错提示或尝试用 future
        # 对于 Streamlit，通常 asyncio.run() 是安全的，除非使用了其他异步库
        # 这里简化处理：直接返回 future 的结果，或者建议安装 nest_asyncio
        # 为了不引入新依赖，我们假设是在标准 Streamlit 环境下，通常会进入下面的 else
        return loop.run_until_complete(coroutine)
    else:
        # 标准 Streamlit 脚本执行路径
        return asyncio.run(coroutine)

def is_self_contained_query(query: str) -> bool:
    """
    Check if query can be answered using LLM's own knowledge
    These queries don't need web search context
    
    Returns True if LLM can answer directly
    """
    query_lower = query.lower()
    
    # Math and logic questions
    math_indicators = [
        'calculate', 'compute', 'multiply', 'divide', 'add', 'subtract',
        'what is', 'equals', '=', 'plus', 'minus', 'times',
        'is', 'are', 'can', 'could', 'would'
    ]
    
    # Self-contained query indicators
    self_contained = [
        # Basic math
        r'\d+\s*[\+\-\*/]\s*\d+',  # "15 * 24"
        r'what is \d+',  # "what is 15"
        
        # General knowledge that doesn't change
        'capital of',
        'definition of',
        'how to',
        'what does .* mean',
        'synonym for',
        'opposite of',
        'formula for',
        'difference between',
        'similarity between'
    ]
    
    import re
    
    # Check for mathematical expressions
    if any(re.search(pattern, query_lower) for pattern in self_contained):
        return True
    
    # Check for math indicators with numbers
    has_numbers = any(char.isdigit() for char in query)
    if has_numbers and any(ind in query_lower for ind in math_indicators):
        return True
    
    return False


def is_factual_query(query: str) -> bool:
    """
    Check if query requires current/factual web information
    Returns True if web search is strongly recommended
    """
    query_lower = query.lower()
    
    factual_indicators = [
        # Time-based indicators
        'latest', 'current', 'recent', 'today', 'this week', 'this month',
        'last 5 days', 'last week', 'last month', 'yesterday',
        'now', 'right now', 'currently',
        
        # News and events
        'news', 'breaking', 'happening',
        
        # Weather
        'weather', 'temperature', 'forecast',
        
        # Years
        '2025', '2024', 'this year',
        
        # Status queries
        'how is', 'what is happening',
        'status of', 'update on',
        
        # Transportation and schedules
        'departure', 'arrival', 'schedule', 'timetable', 'bus', 'train', 'flight',
        'transit', 'transport', 'route', 'station', 'stop',
        
        # Financial/Stock data
        'stock', 'price', 'performance', 'trading', 'market', 'share price',
        'compare.*stock', 'stock performance', 'stock price',
        
        # Real-time data queries
        'live', 'real-time', 'real time', 'up to date', 'current price',
        'current schedule', 'current status'
    ]
    
    return any(ind in query_lower for ind in factual_indicators)


def answer_image_question_directly(image_filename: str, question: str) -> str:
    """
    Answer a question about an uploaded image using GPT Vision API directly
    This is more accurate than using stored descriptions
    """
    try:
        # Get the image file from session state
        if image_filename not in st.session_state.uploaded_image_files:
            return "Image file not found. Please re-upload the image."
        
        image_bytes = st.session_state.uploaded_image_files[image_filename]
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"
        
        # Use GPT Vision API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API key not found."
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        logger.info(f"GPT Vision answered question about {image_filename}")
        return answer
        
    except Exception as e:
        logger.error(f"Error answering image question directly: {e}")
        return f"Error analyzing image: {str(e)}"


def get_uploaded_image_filename(query: str) -> Optional[str]:
    """
    Check if query is about an uploaded image and return the filename
    Returns None if no image query detected or multiple images uploaded
    
    IMPROVED: Detects queries about things in images even without explicit "image" keywords
    """
    if not st.session_state.uploaded_image_files:
        return None
    
    query_lower = query.lower()
    
    # Check if query mentions image/picture/photo explicitly
    image_keywords = ['image', 'picture', 'photo', 'photograph', 'img']
    has_explicit_image_keyword = any(kw in query_lower for kw in image_keywords)
    
    # IMPROVED: Also detect queries about things that are likely in uploaded images
    # These queries suggest the user is asking about content visible in an image
    content_query_patterns = [
        'identify', 'what is this', 'what is that', 'what is the',
        'describe', 'explain', 'tell me about', 'show me',
        'where is', 'where is the', 'where is this', 'where is that',
        'what does this', 'what does that', 'what does the',
        'symbolic', 'meaning', 'significance', 'represents',
        'located', 'location', 'position', 'place'
    ]
    is_content_query = any(pattern in query_lower for pattern in content_query_patterns)
    
    # If query is about identifying/describing something AND we have uploaded images
    # AND the query doesn't explicitly mention documents/files → likely about image
    document_keywords = ['document', 'file', 'pdf', 'uploaded', 'my document', 'the document']
    has_document_keyword = any(kw in query_lower for kw in document_keywords)
    
    # If only one image uploaded and query is about content (not documents) → use image
    if len(st.session_state.uploaded_image_files) == 1:
        if has_explicit_image_keyword or (is_content_query and not has_document_keyword):
            return list(st.session_state.uploaded_image_files.keys())[0]
    
    # If multiple images, try to match by filename mentioned in query
    for filename in st.session_state.uploaded_image_files.keys():
        filename_lower = filename.lower()
        filename_base = filename_lower.rsplit('.', 1)[0]
        if filename_base in query_lower or filename_lower in query_lower:
            return filename
    
    # If query explicitly mentions image/picture and we have images → use first one
    if has_explicit_image_keyword and st.session_state.uploaded_image_files:
        return list(st.session_state.uploaded_image_files.keys())[0]
    
    # If query is about identifying/describing content and we have exactly one image
    # and query doesn't mention documents → likely about the image
    if (is_content_query and not has_document_keyword and 
        len(st.session_state.uploaded_image_files) == 1):
        return list(st.session_state.uploaded_image_files.keys())[0]
    
    return None


def generate_response_with_context(prompt: str, routing_result: Dict) -> tuple[str, float]:
    """
    IMPROVED: Generate response with smart context usage
    - Uses web search context when necessary
    - Allows LLM knowledge for self-contained questions
    - Avoids redundant preambles
    - Includes conversation history for context
    """
    try:
        # Start timing for LLM generation
        llm_start_time = time.time()
        
        llm = ChatOpenAI(
            model="gpt-4o",  # Using GPT-4o for highest quality text completion
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Get routing info
        datasource = routing_result['routing']['datasource']
        context = routing_result.get('context', '')
        retrieval_type = routing_result.get('retrieval_type', 'unknown')
        
        # Get conversation history (last 6 messages for context)
        conversation_history = []
        if 'messages' in st.session_state and len(st.session_state.messages) > 0:
            # Get last 6 messages (3 user + 3 assistant pairs)
            recent_messages = st.session_state.messages[-6:]
            for msg in recent_messages:
                if msg.get('role') in ['user', 'assistant']:
                    conversation_history.append({
                        "role": msg['role'],
                        "content": msg.get('content', '')
                    })
        
        # IMPROVED LOGIC: Check if this is a self-contained query
        is_self_contained = is_self_contained_query(prompt)
        is_factual = is_factual_query(prompt)
        
        logger.info(f"Query analysis: self_contained={is_self_contained}, factual={is_factual}")
        
        # CRITICAL: If it's a factual query (needs real-time data), always use web search context
        # Don't allow self-contained logic to override factual queries
        if is_factual and (datasource == 'web_search' or retrieval_type == 'web_search'):
            # Force use of web search context for factual queries
            is_self_contained = False
        
        # Determine which system prompt to use
        if is_self_contained and not is_factual:
            # SELF-CONTAINED QUERY: Allow LLM to use own knowledge
            # No forced web search context
            logger.info("Using LLM knowledge (self-contained query)")
            
            system_prompt = f"""You are a helpful, knowledgeable assistant.

For this query, provide a clear, direct answer.
- If it's a math problem, solve it step-by-step
- If it's a definition or explanation, provide clear information
- If it's logic/reasoning, use your knowledge

Be concise and direct. No need to cite sources for fundamental knowledge."""
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            # Add conversation history (excluding current prompt)
            for hist_msg in conversation_history[:-1] if len(conversation_history) > 0 and conversation_history[-1].get('content') == prompt else conversation_history:
                messages.append(hist_msg)
            messages.append({"role": "user", "content": f"Question: {prompt}"})
        
        elif datasource == 'web_search' or retrieval_type == 'web_search':
            # WEB SEARCH QUERY: Use the fetched context
            logger.info("Using web search context")
            
            current_date = datetime.now().strftime('%B %d, %Y')
            
            # If context is empty or very short, the web search might have failed
            if not context or len(context.strip()) < 50:
                logger.warning("Web search context is empty or too short, but proceeding with web search prompt")
                system_prompt = f"""You are a helpful assistant that needs to search the web for current information.

The user is asking about real-time or current information that requires web search.

CURRENT DATE: {current_date}

IMPORTANT: This query requires current/real-time data. You should acknowledge that you need to search the web for this information, but you can provide general guidance on:
- Where to find this information (official websites, apps, etc.)
- What type of information would be needed
- General context about the topic

However, if you have access to web search results, use them to provide specific answers."""
            else:
                # Include conversation history context in system prompt
                history_context = ""
                if conversation_history:
                    history_context = "\n\nCONVERSATION HISTORY (for context):\n"
                    for hist_msg in conversation_history[-4:]:  # Last 4 messages
                        role = hist_msg.get('role', 'user')
                        content = hist_msg.get('content', '')
                        if role == 'user':
                            history_context += f"User: {content}\n"
                        else:
                            history_context += f"Assistant: {content}\n"
                    history_context += "\nIMPORTANT: If the user's question is vague or refers to previous conversation (e.g., 'show me all', 'what about that', 'tell me more'), use the conversation history to understand what they're referring to.\n"
                
                # Check if query asks for "next", "earliest", "upcoming", or date-related information
                query_lower = prompt.lower()
                is_date_query = any(keyword in query_lower for keyword in [
                    'next', 'earliest', 'upcoming', 'soonest', 'first', 'nearest',
                    'schedule', 'when is', 'what date', 'what time'
                ])
                
                # Check if query asks for historical/multi-year data
                is_historical_query = any(keyword in query_lower for keyword in [
                    'past', 'over the past', 'last', 'years', 'year by year', 'historical',
                    'compare.*year', 'ranking.*year', 'trend', 'evolution', 'change over time',
                    'ten years', 'five years', 'decade'
                ])
                
                date_instructions = ""
                if is_date_query:
                    date_instructions = "\n\nCRITICAL FOR DATE/TIME QUERIES:\n- If the query asks for 'next', 'earliest', 'upcoming', or 'first' event, you MUST find the EARLIEST date among all search results\n- Compare ALL dates mentioned in the search results and select the one that is CLOSEST to the current date\n- Do NOT just use the first result - check ALL results for dates\n- If multiple events are mentioned, list them in chronological order (earliest first)\n- Current date is: " + current_date + "\n"
                
                historical_instructions = ""
                if is_historical_query:
                    historical_instructions = f"\n\nCRITICAL FOR HISTORICAL/MULTI-YEAR QUERIES:\n- The user is asking for data over MULTIPLE YEARS (e.g., 'past ten years', 'over the past 5 years')\n- You MUST extract data for EACH YEAR mentioned in the search results\n- Organize the information by YEAR in chronological order (oldest to newest)\n- Create a clear comparison showing how values changed year by year\n- If the query asks to 'compare' two entities over time, create a side-by-side comparison by year\n- Present the data in a structured format (table or year-by-year list)\n- DO NOT just mention one or two years - extract ALL years available in the search results\n- If search results mention rankings/values for multiple years, include ALL of them\n- Current date is: {current_date}\n- Example format for rankings:\n  Year | Entity A | Entity B\n  2024 | Rank X  | Rank Y\n  2023 | Rank X  | Rank Y\n  ...\n"
                
                system_prompt = f"""You are a helpful assistant with access to current web search results.{history_context}

Based on the search results provided below, answer the user's question with SPECIFIC information from the results.

CURRENT DATE: {current_date}
{date_instructions}
{historical_instructions}
WEB SEARCH RESULTS:
{context}

CRITICAL INSTRUCTIONS:
1. **ALWAYS USE SEARCH RESULTS FIRST**: If the search results contain the answer, you MUST use that information directly. DO NOT give generic advice like "check the website" or "refer to the official source" - PROVIDE THE ACTUAL INFORMATION from the search results.

2. **Direct Answers Priority**: If you see "Direct Answer" or "OpenAI Web Search Result" in the search results, that information is highly reliable and should be used as the PRIMARY answer. Extract and present that information directly.

3. **Specific Data Extraction**: Extract and use SPECIFIC data from the search results:
   - Dates, times, numbers, statistics
   - Current status, conditions, forecasts
   - Names, locations, rankings
   - Any factual information mentioned

4. **Answer Format**: 
   - Start with the direct answer from search results
   - Use the exact information provided (e.g., "As of [date], [specific fact]")
   - Only add context or explanation if needed for clarity
   - DO NOT say "you would need to check" if the information is already in the search results

5. **When Information is Available**: If search results contain the answer:
   - State it directly: "Based on the latest information, [specific answer]"
   - Include relevant details: dates, numbers, conditions
   - Cite the source if multiple sources confirm

6. **When Information is Missing**: Only if search results don't contain the answer:
   - Acknowledge what information is available
   - Suggest where to find missing information
   - Use general knowledge as last resort

7. **For Historical/Multi-Year Queries**: Extract ALL years mentioned in search results and organize them chronologically. Do not skip years - include every year for which data is available.

8. **Conversation Context**: If the user's question is vague or refers to previous conversation (e.g., "show me all", "what about that"), use the conversation history to understand the context.

REMEMBER: The search results are REAL-TIME and ACCURATE. Use them directly - don't defer to checking websites when the information is already provided."""
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            # Add conversation history (excluding current prompt)
            for hist_msg in conversation_history[:-1] if len(conversation_history) > 0 and conversation_history[-1].get('content') == prompt else conversation_history:
                messages.append(hist_msg)
            messages.append({"role": "user", "content": f"Question: {prompt}"})
        
        elif datasource == 'local_rag':
            # LOCAL DOCUMENT QUERY: Use document context
            logger.info("Using local document context")
            
            system_prompt = f"""You are a helpful assistant with access to uploaded documents.

Based on the document content provided below, answer the user's question.

DOCUMENT CONTENT:
{context}

Rules:
- Answer based on the document content
- If information is not in the documents, clearly state that
- Cite which document you're referencing
- Be clear and concise"""
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            # Add conversation history (excluding current prompt)
            for hist_msg in conversation_history[:-1] if len(conversation_history) > 0 and conversation_history[-1].get('content') == prompt else conversation_history:
                messages.append(hist_msg)
            messages.append({"role": "user", "content": f"Question: {prompt}"})
        
        else:  # hybrid
            # HYBRID QUERY: Combine sources
            logger.info("Using hybrid context (local + web)")
            
            system_prompt = f"""You are a helpful assistant with access to both uploaded documents and web search results.

Use the provided sources to answer comprehensively:

SOURCES:
{context}

Rules:
- Use both local documents and web information
- Distinguish between document content and web information
- Be clear about sources
- Provide comprehensive answer"""
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            # Add conversation history (excluding current prompt)
            for hist_msg in conversation_history[:-1] if len(conversation_history) > 0 and conversation_history[-1].get('content') == prompt else conversation_history:
                messages.append(hist_msg)
            messages.append({"role": "user", "content": f"Question: {prompt}"})
        
        # Generate response
        logger.info(f"Generating response for: {prompt[:50]}...")
        response = llm.invoke(messages)
        answer = response.content
        
        # Calculate LLM generation time
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time
        logger.info(f"⏱️ LLM Generation Time: {llm_duration:.2f} seconds")
        
        return answer, llm_duration

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}", 0.0


def main():
    st.title("🤖 Enhanced NLP RAG Chatbot with Live Web Search")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload (Optional)")
        st.info("Upload PDFs or images to enable document search")

        uploaded_files = st.file_uploader(
            "Upload PDFs or Images",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🔄 Process Files"):
                vectorstore = process_uploaded_files(uploaded_files)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.rerun()

        st.divider()

        # Display processed files
        if st.session_state.uploaded_files_processed:
            st.subheader("✅ Processed Files")
            for file_name in st.session_state.uploaded_files_processed:
                st.text(f"📄 {file_name}")

        st.divider()

        # Routing statistics
        if st.session_state.source_router:
            stats = st.session_state.source_router.get_routing_stats()
            if stats['total_queries'] > 0:
                st.subheader("📊 Routing Stats")
                st.metric("Total Queries", stats['total_queries'])

                col1, col2 = st.columns(2)
                with col1:
                    for source, count in stats['by_source'].items():
                        st.metric(source.replace('_', ' ').title(), count)
                with col2:
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    st.metric("Success Rate", f"{stats['success_rate']:.1%}")

        st.divider()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize router
    initialize_source_router()

    # Chat interface
    st.header("💬 Ask Anything")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display performance metrics if available (always show for assistant messages)
            if message["role"] == "assistant" and ("search_duration" in message or "llm_duration" in message):
                search_duration = message.get("search_duration", 0.0)
                llm_duration = message.get("llm_duration", 0.0)
                total_time = search_duration + llm_duration
                
                with st.expander("⏱️ Performance Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Search/Routing Time", f"{search_duration:.2f}s")
                    with col2:
                        st.metric("LLM Generation Time", f"{llm_duration:.2f}s")
                    
                    st.metric("Total Response Time", f"{total_time:.2f}s")

            # Display routing info if available
            if "routing" in message:
                with st.expander("🔍 Source Information"):
                    routing = message["routing"]
                    source_name = routing['datasource'].replace('_', ' ').title()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Source", source_name)
                    with col2:
                        st.metric("Confidence", f"{routing['confidence']:.1%}")
                    with col3:
                        st.metric("Type", message.get('retrieval_type', 'unknown').replace('_', ' ').title())

                    st.info(f"**Reasoning**: {routing['reasoning']}")

            # Display sources
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 {len(message['sources'])} Sources Used"):
                    for i, source in enumerate(message["sources"][:10], 1):
                        if source.get('type') == 'web':
                            st.markdown(f"**{i}. 🌐 {source.get('title', 'Web Source')}**")
                            if source.get('url'):
                                st.caption(f"URL: {source['url']}")
                        else:
                            st.markdown(f"**{i}. 📄 {source.get('source', 'Local Document')}**")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if this is a direct image question - use GPT Vision API directly
                    image_filename = get_uploaded_image_filename(prompt)
                    if image_filename:
                        logger.info(f"Detected image query, using GPT Vision API directly for: {image_filename}")
                        
                        # Time the image processing
                        image_start_time = time.time()
                        answer = answer_image_question_directly(image_filename, prompt)
                        image_end_time = time.time()
                        image_duration = image_end_time - image_start_time
                        
                        st.markdown(answer)
                        
                        # Display sources for image query
                        with st.expander(f"📚 1 Source Used"):
                            st.markdown(f"**1. 🖼️ {image_filename}**")
                            st.caption("Image analyzed using GPT Vision API")
                        
                        # Display performance metrics for image query
                        with st.expander("⏱️ Performance Metrics"):
                            st.metric("Image Processing Time", f"{image_duration:.2f}s")
                            st.metric("Search/Routing Time", "0.00s")
                            st.metric("LLM Generation Time", f"{image_duration:.2f}s")
                            st.metric("Total Response Time", f"{image_duration:.2f}s")
                        
                        # Save assistant message with performance metrics
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "routing": {"datasource": "image_vision", "reasoning": f"Direct GPT Vision analysis of {image_filename}", "confidence": 1.0},
                            "sources": [{"type": "image", "source": image_filename}],
                            "retrieval_type": "image_vision",
                            "search_duration": 0.0,
                            "llm_duration": image_duration
                        })
                    else:
                        # Start timing for search/routing processing
                        search_start_time = time.time()
                        
                        # Get retriever if available
                        retriever = None
                        has_docs = st.session_state.vectorstore is not None

                        if has_docs:
                            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})

                        # Route the query (this includes search processing)
                        routing_result = run_async(
                            st.session_state.source_router.route_query(
                                query=prompt,
                                local_retriever=retriever,
                                has_uploaded_docs=has_docs
                            )
                        )
                        
                        # End timing for search/routing processing
                        search_end_time = time.time()
                        search_duration = search_end_time - search_start_time
                        logger.info(f"⏱️ Search/Routing Processing Time: {search_duration:.2f} seconds")

                        logger.info(f"Query routed to: {routing_result['routing']['datasource']}")

                        # Generate response with improved logic (includes LLM timing)
                        answer, llm_duration = generate_response_with_context(prompt, routing_result)

                        st.markdown(answer)
                        
                        # Display sources if available
                        if routing_result.get('sources'):
                            with st.expander(f"📚 {len(routing_result.get('sources', []))} Sources Used"):
                                for i, source in enumerate(routing_result['sources'][:10], 1):
                                    if source.get('type') == 'web':
                                        st.markdown(f"**{i}. 🌐 {source.get('title', 'Web Source')}**")
                                        if source.get('url'):
                                            st.caption(f"URL: {source['url']}")
                                        if source.get('snippet'):
                                            st.caption(f"{source['snippet'][:200]}...")
                                    else:
                                        st.markdown(f"**{i}. 📄 {source.get('source', 'Local Document')}**")
                                        if source.get('page'):
                                            st.caption(f"Page: {source.get('page')}")
                                        if source.get('content'):
                                            st.caption(f"{source['content'][:200]}...")
                        
                        # Display timing information
                        with st.expander("⏱️ Performance Metrics"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Search/Routing Time", f"{search_duration:.2f}s")
                            with col2:
                                st.metric("LLM Generation Time", f"{llm_duration:.2f}s")
                            
                            total_time = search_duration + llm_duration
                            st.metric("Total Response Time", f"{total_time:.2f}s")

                        # Show routing details
                        with st.expander("🔍 Full Routing Details"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Routing Decision")
                                routing = routing_result['routing']
                                st.write(f"**Data Source**: {routing['datasource'].replace('_', ' ').title()}")
                                st.write(f"**Reasoning**: {routing['reasoning']}")
                                st.write(f"**Confidence**: {routing['confidence']:.1%}")

                            with col2:
                                st.subheader("Search Results")
                                retrieval_type = routing_result.get('retrieval_type', 'unknown')
                                st.write(f"**Retrieval Type**: {retrieval_type.replace('_', ' ').title()}")
                                st.write(f"**Results Found**: {len(routing_result.get('sources', []))}")

                                if routing_result.get('num_web_results'):
                                    st.write(f"**Web Results**: {routing_result['num_web_results']}")
                                if routing_result.get('num_local_results'):
                                    st.write(f"**Local Results**: {routing_result['num_local_results']}")

                            # Show full context preview
                            st.subheader("Retrieved Context (Preview)")
                            context_preview = routing_result.get('context', '')[:1000]
                            st.text_area("Context:", value=context_preview + ("..." if len(routing_result.get('context', '')) > 1000 else ""), height=200, disabled=True)

                        # Save assistant message with routing info and performance metrics
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "routing": routing_result['routing'],
                            "sources": routing_result.get('sources', []),
                            "retrieval_type": routing_result.get('retrieval_type', 'unknown'),
                            "search_duration": search_duration,
                            "llm_duration": llm_duration
                        })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Response generation error: {e}", exc_info=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "routing": {"datasource": "error", "reasoning": str(e), "confidence": 0},
                        "sources": [],
                        "search_duration": 0.0,
                        "llm_duration": 0.0
                    })


if __name__ == "__main__":
    main()
