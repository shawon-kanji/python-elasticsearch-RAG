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

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es_client = AsyncElasticsearch(
    os.getenv("ELASTICSEARCH_URL"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embedding mode
# Anthropic LLM via LangChain
llm = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20240620",
    temperature=0
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to do, client is already initialized
    yield
    # Shutdown: cleanup
    await es_client.close()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)


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

        # Step 3: Load embedding model
        embedder = SentenceTransformer(MODEL_NAME)

        # Step 4: Prepare bulk actions
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
                        "embeddings": embedding
                    }
                }

        # Step 5: Run bulk indexing
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
async def search_context(query: str, k: int = 3):
    try:
        # Step 1: Load embedder
        embedder = SentenceTransformer(MODEL_NAME)
        query_vector = embedder.encode(query).tolist()

        # Step 2: Run kNN vector search in ES
        response = await es_client.search(
            index="rag_documents",
            knn={
                "field": "embeddings",        # must match your mapping field
                "query_vector": query_vector,
                "num_candidates": 100         # more candidates → better recall
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


@app.get("/rag/query")
async def rag_query(query: str = Query(...), k: int = 5):
    try:
        embedder = SentenceTransformer(MODEL_NAME)
        query_vector = embedder.encode(query).tolist()
        response = await es_client.search(
            index="rag_documents",
            knn={
                "field": "embeddings",
                "query_vector": query_vector,
                "num_candidates": 100
            },
            source=["content", "source"]
        )
        contexts = [
            {
                "score": hit["_score"],
                "content": hit["_source"]["content"],
                "source": hit["_source"]["source"]
            }
            for hit in response["hits"]["hits"][:k]
        ]

        # Prepare the prompt with retrieved contexts
        context_text = "\n\n".join([c["content"] for c in contexts])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant for internal documents. Use ONLY the provided context to answer. Do not add anything beyond the context."),
            ("human",
             "Question: {question}\n\nContext:\n{context}\n\nAnswer in detail:")
        ])

        chain = prompt | llm
        result = chain.invoke({"question": query, "context": context_text})
        return {
            "status": "success",
            "query": query,
            "answer": result,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
