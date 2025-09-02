from contextlib import asynccontextmanager
from fastapi import FastAPI
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
es_client = AsyncElasticsearch(
    os.getenv("ELASTICSEARCH_URL"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
