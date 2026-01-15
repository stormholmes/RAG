# Contributions

## Summary

**Jason** - (Codebase, Enhanced search architecture by adding GPT Vision API integration and OpenAI Web Search API integration, Improved LLM model usage with GPT-4o, Enhanced similarity search with Cross-Encoder reranking, Better prompt templates with conversation history and date-aware instructions, Multi-source search combining OpenAI + Tavily, Conversation memory for follow-up questions, Intelligent query processing with multi-part detection and targeted searches)

## Key Contributions

- **LLM Model:** Upgraded to GPT-4o for text completion and GPT Vision for image analysis
- **Better Similarity Search:** Implemented Cross-Encoder reranking (MS MARCO model) with BM25 fallback
- **Better Prompt Template:** Enhanced prompts with conversation history, date-aware instructions, and context-aware routing
- **OpenAI Web Search API Integration:** Real-time query handling for bus schedules, stock prices, weather, events
- **Enhanced Search Architecture:** Multi-source search with fallback chain (OpenAI → Tavily → Wikipedia → ArXiv → Google → Bing)
- **Conversation Memory & Context:** Tracks last 6 messages for follow-up question understanding
- **Intelligent Query Processing:** Multi-part query detection, targeted searches, date-aware handling
