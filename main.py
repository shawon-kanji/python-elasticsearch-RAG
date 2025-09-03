import uuid
from elasticsearch.helpers import async_bulk
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es_client = AsyncElasticsearch(
    os.getenv("ELASTICSEARCH_URL"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
# Anthropic LLM via LangChain
llm = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20240620",
    temperature=0
)

# Initialize embedding model globally
embedder = SentenceTransformer(MODEL_NAME)


def embed_func(texts):
    if isinstance(texts, str):
        return embedder.encode(texts).tolist()
    return [embedder.encode(t).tolist() for t in texts]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to do, client is already initialized
    yield
    # Shutdown: cleanup
    await es_client.close()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Store conversation memories in-memory (you could also store in ES)
conversation_memories = {}


@app.get("/es/health")
async def health_check():
    try:
        health = await es_client.cluster.health()
        return {
            "status": "healthy",
            "elasticsearch_status": health["status"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/es/ingest")
async def ingest_data():
    DATA_FILE = os.path.join(os.path.dirname(
        __file__), "datasource", "story1.md")
    try:
        # Step 1: Load markdown file
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        # Step 2: Chunk the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # Step 3: Prepare bulk actions
        async def gen_actions():
            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                yield {
                    "_op_type": "index",         # index or create/update
                    "_index": "rag_documents",   # target index
                    "_id": str(uuid.uuid4()),    # unique ID
                    "_source": {
                        "id": i,
                        "content": chunk,
                        "source": "story1.md",
                        "embeddings": embedding  # Fixed: keep as "embeddings"
                    }
                }

        # Step 4: Run bulk indexing
        success, failed = await async_bulk(es_client, gen_actions())
        await es_client.indices.refresh(index="rag_documents")

        return {
            "status": "ingestion complete",
            "chunks_indexed": success,
            "failed": failed
        }

    except Exception as e:
        return {
            "status": "ingestion failed",
            "error": str(e)
        }


@app.get("/es/search")
async def search_context(query: str, k: int = 8):
    try:
        # Step 1: Get query embedding
        query_vector = embedder.encode(query).tolist()

        # Step 2: Run kNN vector search in ES
        response = await es_client.search(
            index="rag_documents",
            knn={
                "field": "embeddings",        # must match your mapping field
                "query_vector": query_vector,
                "k": k,                       # Fixed: use k instead of num_candidates
                "num_candidates": k * 10      # more candidates → better recall
            },
            source=["content", "source"]     # only return relevant fields
        )

        print("context::: ", response)

        # Step 3: Extract hits
        contexts = [
            {
                "score": hit["_score"],
                "content": hit["_source"]["content"],
                "source": hit["_source"]["source"]
            }
            for hit in response["hits"]["hits"]
        ]

        return {
            "status": "success",
            "query": query,
            "results": contexts
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


# ---------- Conversational RAG ----------
@app.get("/rag/query")
async def rag_query(query: str, k: int = 5, thread_id: str = None):
    """
    Conversational RAG with in-memory conversation history:
    - Retrieves relevant story context from ES
    - Stores chat history in memory
    """
    try:
        # Step 1: If new conversation, create a new thread
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Step 2: Get or create memory for this thread
        if thread_id not in conversation_memories:
            conversation_memories[thread_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        memory = conversation_memories[thread_id]

        # Step 3: Embed query and search in story index
        query_vector = embedder.encode(query).tolist()
        response = await es_client.search(
            index="rag_documents",
            knn={
                "field": "embeddings",        # Fixed: match the field name from ingestion
                "query_vector": query_vector,
                "k": k,
                "num_candidates": k * 10
            },
            source=["content", "source"]
        )
        contexts = [hit["_source"]["content"]
                    for hit in response["hits"]["hits"]]

        # Step 4: Get chat history
        chat_history = memory.chat_memory.messages
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"Human: {msg.content}" if isinstance(msg, HumanMessage)
                else f"Assistant: {msg.content}"
                for msg in chat_history[-6:]  # Last 3 exchanges
            ])

        # Step 5: Create prompt template
        prompt_template = """You are a helpful AI assistant. Use the conversation history and retrieved context to answer the question.

                            Context from documents:
                            {context}

                            Conversation History:
                            {history}

                            Current Question: {question}

                            Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Step 6: Format the prompt
        formatted_prompt = prompt.format(
            context="\n\n".join(contexts),
            history=history_text,
            question=query
        )

        # Step 7: Get response from LLM
        response = await llm.ainvoke([HumanMessage(content=formatted_prompt)])
        answer = response.content

        # Step 8: Update memory
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(answer)

        return {
            "status": "success",
            "thread_id": thread_id,
            "query": query,
            "answer": answer,
            "contexts_used": len(contexts)
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}


@app.get("/rag/clear_memory")
async def clear_memory(thread_id: str):
    """Clear conversation memory for a specific thread"""
    if thread_id in conversation_memories:
        del conversation_memories[thread_id]
        return {"status": "success", "message": f"Memory cleared for thread {thread_id}"}
    else:
        return {"status": "not_found", "message": f"Thread {thread_id} not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
