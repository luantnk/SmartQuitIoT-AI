import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "AI Service is ready"}


@patch("app.main.is_text_toxic")
def test_check_text_toxic(mock_is_toxic):

    mock_is_toxic.return_value = True

    response = client.post("/check-content", json={"text": "You are stupid"})

    assert response.status_code == 200
    assert response.json()["isToxic"] is True
    assert response.json()["type"] == "text"


@patch("app.main.is_text_toxic")
def test_check_text_safe(mock_is_toxic):
    mock_is_toxic.return_value = False

    response = client.post("/check-content", json={"text": "Hello friend"})

    assert response.status_code == 200
    assert response.json()["isToxic"] is False


@patch("app.main.check_image_url")
def test_check_image_safe(mock_check_image):
    mock_check_image.return_value = False

    response = client.post("/check-image-url", data={"image_url": "http://example.com/image.jpg"})

    assert response.status_code == 200
    assert response.json()["type"] == "image"
    assert response.json()["isToxic"] is False


@patch("app.main.check_image_url")
def test_check_image_error(mock_check_image):
    mock_check_image.side_effect = ValueError("Invalid Image Format")

    response = client.post("/check-image-url", data={"image_url": "bad_url"})

    assert response.status_code == 422
    assert "Invalid Image" in response.json()["detail"]