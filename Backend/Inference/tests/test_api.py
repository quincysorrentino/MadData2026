from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (224, 224), color=color)
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_infer_image_file() -> None:
    payload = _image_bytes((30, 30, 30))
    response = client.post(
        "/v1/infer",
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["label"], str)
    assert body["label"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["boxes"], list)
    assert isinstance(body["model_version"], str)
    assert body["model_version"]


def test_infer_reject_non_image() -> None:
    response = client.post(
        "/v1/infer",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_infer_alias_route() -> None:
    payload = _image_bytes((200, 200, 200))
    response = client.post(
        "/infer",
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert response.status_code == 200
    assert isinstance(response.json()["boxes"], list)


def test_infer_debug_route() -> None:
    payload = _image_bytes((120, 120, 120))
    response = client.post(
        "/infer/debug",
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["label"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["boxes"], list)
    assert isinstance(body["debug"], dict)
    assert "decision_reason" in body["debug"]
