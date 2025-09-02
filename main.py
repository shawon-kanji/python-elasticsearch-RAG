import uuid
from elasticsearch.helpers import async_bulk
from contextlib import asynccontextmanager
from fastapi import FastAPI
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es_client = AsyncElasticsearch(
    os.getenv("ELASTICSEARCH_URL"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embedding mode


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
