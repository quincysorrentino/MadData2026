from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import ALLOWED_ORIGINS
import services.prompt_service as prompt_service
from routers import body_part, diagnosis, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    prompt_service.load_system_prompts()
    yield


app = FastAPI(
    title="Skin Cancer Classifier API",
    description=(
        "Backend service for the skin cancer classification app. "
        "Bridges the frontend, a skin lesion classifier, and a locally-running Mistral LLM."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(body_part.router, prefix="/api")
app.include_router(diagnosis.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
