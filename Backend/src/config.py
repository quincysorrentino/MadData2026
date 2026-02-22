import os


def _parse_origins(raw_value: str) -> list[str]:
	return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def _parse_int(raw_value: str, default: int) -> int:
	try:
		return int(raw_value)
	except (TypeError, ValueError):
		return default


def _parse_positive_int(raw_value: str, default: int) -> int:
	value = _parse_int(raw_value, default)
	return value if value > 0 else default


CLASSIFIER_API_URL = os.getenv("CLASSIFIER_API_URL", "http://127.0.0.1:8000")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:9000")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
LLM_TIMEOUT_SECONDS = _parse_positive_int(os.getenv("LLM_TIMEOUT_SECONDS", "90"), 90)
LLM_MAX_TOKENS = _parse_positive_int(os.getenv("LLM_MAX_TOKENS", "512"), 512)
LLM_CONNECT_TIMEOUT_SECONDS = _parse_positive_int(
	os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "5"),
	5,
)
LLM_POOL_TIMEOUT_SECONDS = _parse_positive_int(
	os.getenv("LLM_POOL_TIMEOUT_SECONDS", "5"),
	5,
)
LLM_CONVERSATION_MAX_TURNS = _parse_positive_int(
	os.getenv("LLM_CONVERSATION_MAX_TURNS", "6"),
	6,
)
LLM_MAX_MESSAGE_CHARS = _parse_positive_int(
	os.getenv("LLM_MAX_MESSAGE_CHARS", "2000"),
	2000,
)
LLM_MAX_CONCURRENCY = _parse_positive_int(
	os.getenv("LLM_MAX_CONCURRENCY", "1"),
	1,
)
DIAGNOSE_MAX_IMAGE_MB = _parse_positive_int(
	os.getenv("DIAGNOSE_MAX_IMAGE_MB", "8"),
	8,
)
DIAGNOSE_MAX_IMAGE_BYTES = DIAGNOSE_MAX_IMAGE_MB * 1024 * 1024
ALLOWED_ORIGINS = _parse_origins(
	os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
)
