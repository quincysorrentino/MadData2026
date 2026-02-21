import os


def _parse_origins(raw_value: str) -> list[str]:
	return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://127.0.0.1:8000")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
ALLOWED_ORIGINS = _parse_origins(
	os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
)
