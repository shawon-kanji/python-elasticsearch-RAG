# 🧠 Conversational RAG System

A **Retrieval-Augmented Generation (RAG)** system built with FastAPI that combines document search capabilities with conversational memory. This system uses Elasticsearch for document storage and retrieval, and Anthropic's Claude for intelligent responses with conversation context.

## ✨ Features

- **📄 Document Ingestion**: Automatically chunk and embed documents into Elasticsearch
- **🔍 Semantic Search**: Vector similarity search across your document corpus
- **💬 Conversational Memory**: Persistent conversation history across sessions
- **🧠 Context-Aware Responses**: Leverages both document context and conversation history
- **🔧 Memory Management**: Smart conversation thread management


## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   User Query     │    │  Conversation   │
│  (Markdown)     │    │                  │    │    Memory       │
└─────────┬───────┘    └─────────┬────────┘    │  (In-Memory)    │
          │                      │             └─────────────────┘
          ▼                      ▼                      ▲
┌─────────────────┐    ┌──────────────────┐             │
│  Text Splitter  │    │   Embeddings     │             │
│  (LangChain)    │    │                  │             │
└─────────┬───────┘    └─────────┬────────┘             │
          │                      │                      │
          ▼                      ▼                      │
┌─────────────────┐    ┌──────────────────┐             │
│  Elasticsearch  │◄───┤  Vector Search   │             │
│   (Vectors +    │    │     (kNN)        │             │
│   Documents)    │    └─────────┬────────┘             │
└─────────────────┘              │                      │
                                 ▼                      │
                       ┌──────────────────┐             │
                       │   Claude LLM     │─────────────┘
                       │   (Anthropic)    │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Elasticsearch cluster (local or cloud)
- Anthropic API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd conversational-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your configuration:
   ```env
   ELASTICSEARCH_URL=http://localhost:9200
   ELASTICSEARCH_API_KEY=your_elasticsearch_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

4. **Prepare your documents**
   ```bash
   mkdir datasource
   # Add your markdown files to the datasource directory
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📚 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Core Endpoints

#### Health Check
```bash
GET /es/health
```
Check Elasticsearch connectivity and system health.

#### Document Ingestion
```bash
GET /es/ingest
```
Process and index documents from the `datasource` directory.

#### Document Search
```bash
GET /es/search?query=your_question&k=5
```
Search documents without conversation context.

#### Conversational RAG
```bash
GET /rag/query?query=your_question&thread_id=optional_thread_id&k=5
```
Query with conversation memory. If no `thread_id` is provided, a new conversation thread is created.

#### Memory Management
```bash
GET /rag/clear_memory?thread_id=your_thread_id
```
Clear conversation history for a specific thread.

## 🔧 Configuration

### Document Processing
- **Chunk Size**: 500 characters (configurable in `RecursiveCharacterTextSplitter`)
- **Chunk Overlap**: 50 characters
- **Separators**: `["\n\n", "\n", " ", ""]`

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Use Case**: Balanced performance and quality for semantic search

### LLM Configuration
- **Model**: `claude-3-5-sonnet-20240620`
- **Temperature**: 0 (deterministic responses)
- **Provider**: Anthropic via LangChain

### Memory Settings
- **Type**: ConversationBufferMemory
- **History Limit**: Last 6 messages (3 exchanges)
- **Storage**: In-memory (resets on restart)

## 🔄 Usage Examples

### Basic Document Query
```bash
curl "http://localhost:8000/rag/query?query=What is the main topic of the story?"
```

### Follow-up Question
```bash
curl "http://localhost:8000/rag/query?query=Tell me more about that&thread_id=your_thread_id"
```

### Conversation Flow
```python
import requests

base_url = "http://localhost:8000"

# Start conversation
response1 = requests.get(f"{base_url}/rag/query", params={
    "query": "What are the main characters?"
})
thread_id = response1.json()["thread_id"]

# Follow-up question
response2 = requests.get(f"{base_url}/rag/query", params={
    "query": "What happens to them in the end?",
    "thread_id": thread_id
})

print(response2.json()["answer"])
```

## 📁 Project Structure

```
├── main.py                 # FastAPI application
├── datasource/            # Document storage directory
│   └── story1.md         # Example document
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🛠️ Development

### Adding New Document Types

To support different document formats, extend the ingestion logic in `ingest_data()`:

```python
# Example: PDF support
from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
```

### Custom Embedding Models

Replace the embedding model by updating:

```python
MODEL_NAME = "sentence-transformers/your-preferred-model"
embedder = SentenceTransformer(MODEL_NAME)
```

### Advanced Memory Strategies

For persistent memory across restarts, consider:

- **Database Storage**: PostgreSQL, MongoDB
- **Vector Databases**: ChromaDB, Pinecone, Weaviate
- **Hybrid Approach**: Recent messages in memory + historical in database

## 🚀 Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Production environment variables:

```env
ELASTICSEARCH_URL=https://your-es-cluster.com:9200
ELASTICSEARCH_API_KEY=your_production_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LOG_LEVEL=INFO
```

### Health Monitoring

The `/es/health` endpoint provides system health status for monitoring tools.

## 🔐 Security Considerations

- **API Keys**: Never commit API keys to version control
- **Authentication**: Add API authentication for production use
- **Rate Limiting**: Implement rate limiting for public deployments
- **Input Validation**: Validate and sanitize user inputs
- **CORS**: Configure CORS settings appropriately

## 📈 Performance Tuning

### Elasticsearch Optimization
- **Index Settings**: Adjust shards and replicas based on data size
- **Mapping**: Define explicit mappings for better performance
- **Caching**: Enable query result caching

### Memory Management
- **Conversation Limits**: Implement automatic cleanup of old conversations
- **Memory Monitoring**: Monitor memory usage in production
- **Async Processing**: Use async operations for better concurrency

## 🐛 Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   ```bash
   # Check if Elasticsearch is running
   curl -X GET "localhost:9200/_cluster/health"
   ```

2. **Empty Search Results**
   - Verify documents are ingested: `GET /es/ingest`
   - Check index exists in Elasticsearch
   - Verify embedding model is working

3. **Memory Issues**
   - Restart the application to clear in-memory conversations
   - Check available system memory

4. **API Key Errors**
   - Verify environment variables are loaded
   - Check API key validity with Anthropic

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 Requirements

```txt
fastapi>=0.104.0
uvicorn>=0.24.0
elasticsearch>=8.10.0
python-dotenv>=1.0.0
langchain>=0.0.300
langchain-anthropic>=0.1.0
sentence-transformers>=2.2.2
langchain-text-splitters>=0.0.1
```

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [Elasticsearch](https://www.elastic.co/) - Search and analytics engine
- [Anthropic](https://www.anthropic.com/) - Claude LLM provider
- [Sentence Transformers](https://www.sbert.net/) - Embedding models


---

⭐ **If this project helped you, please give it a star!** ⭐