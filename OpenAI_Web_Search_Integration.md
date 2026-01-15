# OpenAI Web Search Integration Guide

## 📋 Overview

We've integrated **OpenAI's Web Search API** to handle real-time queries that require up-to-date information, such as:
- Bus/train schedules
- Stock prices and financial data
- Weather forecasts
- Current news and events
- Live data queries

## ✅ What We've Implemented

### 1. **New Module: `web_search_openai.py`**

A dedicated integration class for OpenAI's web search capabilities:

- **OpenAIWebSearchIntegration**: Main class for OpenAI web search
- Supports three modes:
  - **Fast search** (`reasoning_level="low"`): Quick lookups
  - **Thorough search** (`reasoning_level="medium"`): Complex queries
  - **Deep research** (`reasoning_level="high"`): Extended investigations

### 2. **Enhanced `web_search_tavily.py`**

Updated `WebSearchEnhanced` class to:
- Automatically detect real-time queries
- Use OpenAI web search first for real-time queries
- Fall back to existing search chain if OpenAI fails

**New Fallback Chain:**
```
OpenAI Web Search (for real-time) → Tavily → Wikipedia → ArXiv → Google API → Bing API → Mock
```

## 🔧 How It Works

### **Automatic Real-Time Query Detection**

The system automatically detects queries that require real-time data:

```python
def _is_real_time_query(self, query: str) -> bool:
    """Detects queries requiring real-time data"""
    real_time_keywords = [
        'departure time', 'bus schedule', 'train schedule', 'flight',
        'stock price', 'stock performance', 'nvidia', 'amd', 'ticker',
        'weather', 'temperature', 'forecast', 'today', 'now', 'current',
        'latest', 'recent', 'breaking', 'news', 'live'
    ]
    return any(keyword in query.lower() for keyword in real_time_keywords)
```

### **Query Flow**

1. **User asks real-time question** (e.g., "What are the departure times for Bus 91M?")
2. **System detects** it's a real-time query
3. **OpenAI Web Search** is used first (if available)
4. **If OpenAI fails**, falls back to Tavily → Wikipedia → etc.
5. **Results returned** with citations and sources

## 🚀 Usage Examples

### **Example 1: Bus Schedule Query**

**Query:** "What are the departure times for the Bus 91M from Diamond Hill station?"

**What happens:**
1. System detects "departure time" and "bus schedule" keywords
2. Routes to OpenAI Web Search first
3. OpenAI searches the web for current bus schedule information
4. Returns real-time schedule data with citations

### **Example 2: Stock Price Query**

**Query:** "Compare the stock performance of NVIDIA (NVDA) and AMD over the last 5 days"

**What happens:**
1. System detects "stock performance" and "nvidia"/"amd" keywords
2. Routes to OpenAI Web Search
3. OpenAI fetches current stock data from financial sources
4. Returns comparison with up-to-date prices and performance metrics

## 📝 API Compatibility

### **Two API Methods Supported:**

1. **Responses API** (Preferred, if available)
   ```python
   response = client.responses.create(
       model="gpt-4o",
       tools=[{"type": "web_search"}],
       input=query
   )
   ```

2. **Chat Completions API** (Fallback)
   ```python
   response = client.chat.completions.create(
       model="gpt-4o-search-preview",  # or gpt-5-search-api
       messages=[{"role": "user", "content": query}]
   )
   ```

The system automatically tries Responses API first, then falls back to Chat Completions if needed.

## ⚙️ Configuration

### **Environment Variables**

Make sure you have your OpenAI API key set:

```bash
OPENAI_API_KEY=your_api_key_here
```

### **Initialization**

The system automatically initializes OpenAI web search when `WebSearchEnhanced` is created:

```python
web_search = WebSearchEnhanced(
    max_results=5,
    timeout=15,
    use_openai_web_search=True  # Enable OpenAI web search
)
```

## 🎯 Benefits

### **Why Use OpenAI Web Search for Real-Time Queries?**

1. **Better Real-Time Data**: OpenAI's web search is optimized for current information
2. **Automatic Source Selection**: OpenAI chooses the best sources for the query
3. **Citations Included**: Results come with source URLs and citations
4. **Contextual Understanding**: Better understanding of what information is needed
5. **No Manual API Setup**: No need to configure multiple APIs (weather, finance, etc.)

### **Comparison with Current System**

| Feature | Current (Tavily) | OpenAI Web Search |
|---------|------------------|-------------------|
| **Real-time data** | ⚠️ May be outdated | ✅ Current/live data |
| **Bus schedules** | ❌ Limited | ✅ Good coverage |
| **Stock prices** | ❌ Limited | ✅ Real-time quotes |
| **Weather** | ⚠️ General | ✅ Current forecasts |
| **Citations** | ✅ Yes | ✅ Yes (inline) |
| **Cost** | ✅ Free tier | ⚠️ API costs |

## 🔍 How to Test

### **Test Real-Time Queries:**

1. **Bus Schedule:**
   ```
   "What are the departure times for the Bus 91M from Diamond Hill station?"
   ```

2. **Stock Prices:**
   ```
   "Compare the stock performance of NVIDIA (NVDA) and AMD over the last 5 days"
   ```

3. **Weather:**
   ```
   "What's the weather forecast for Hong Kong today?"
   ```

### **Expected Behavior:**

- System should detect these as real-time queries
- OpenAI Web Search should be used first
- Results should include current data with citations
- If OpenAI fails, system falls back to Tavily

## 🐛 Troubleshooting

### **Issue: OpenAI Web Search Not Working**

**Possible Causes:**
1. **API Key Missing**: Check `OPENAI_API_KEY` in `.env`
2. **SDK Version**: Ensure `openai>=1.10.0` is installed
3. **API Access**: Responses API may not be available in your region/account

**Solution:**
- The system automatically falls back to Chat Completions API
- If that also fails, it falls back to Tavily search

### **Issue: Not Detecting Real-Time Queries**

**Solution:**
- Check if query contains real-time keywords
- Manually add keywords to `_is_real_time_query()` if needed

## 📊 Integration Points

### **Files Modified:**

1. **`web_search_openai.py`** (NEW)
   - OpenAI Web Search integration class
   - Handles Responses API and Chat Completions fallback

2. **`web_search_tavily.py`** (UPDATED)
   - Added `_is_real_time_query()` method
   - Added `search_openai()` method
   - Updated `search()` to use OpenAI first for real-time queries

3. **`intelligent_source_router.py`** (NO CHANGES NEEDED)
   - Already routes real-time queries to web search
   - Will automatically benefit from OpenAI integration

## 🎓 For Your Presentation

### **Key Points to Highlight:**

1. **Intelligent Routing:**
   - System automatically detects real-time queries
   - Routes to best search method (OpenAI for real-time, Tavily for general)

2. **Multi-Layer Fallback:**
   - OpenAI Web Search → Tavily → Wikipedia → ArXiv → Google → Bing → Mock
   - Guaranteed to return results

3. **Real-Time Capabilities:**
   - Can now answer bus schedules, stock prices, weather, etc.
   - Uses OpenAI's optimized web search for current data

4. **Robust Error Handling:**
   - Gracefully falls back if OpenAI API unavailable
   - System continues working with other search methods

## 🚧 Future Enhancements

### **Potential Improvements:**

1. **Domain Filtering:**
   - Filter results to specific domains (e.g., transit authority websites for bus schedules)
   - Already supported in OpenAI Web Search API

2. **Location-Aware Search:**
   - Use `user_location` parameter for location-specific queries
   - Better results for "near me" queries

3. **Reasoning Level Selection:**
   - Use "medium" or "high" reasoning for complex queries
   - Use "low" for simple lookups (faster)

4. **Caching:**
   - Cache real-time results for a short period
   - Reduce API calls for repeated queries

## ✅ Summary

**What We've Achieved:**

- ✅ Integrated OpenAI Web Search API
- ✅ Automatic real-time query detection
- ✅ Intelligent routing (OpenAI first for real-time, Tavily for general)
- ✅ Robust fallback chain
- ✅ Handles bus schedules, stock prices, weather, etc.
- ✅ Includes citations and sources

**Result:** Your chatbot can now answer real-time questions that require current data! 🎉

