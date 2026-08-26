from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_200_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_cross_origin_requests_from_vite_dev_server():
    """The Phase 8 React dashboard (Vite dev server, http://localhost:5173)
    is a different origin than this API -- without CORS headers, every
    fetch call the frontend makes fails before reaching a route. Asserts
    the actual browser-relevant behavior (an Origin header gets an
    Access-Control-Allow-Origin response back), not just that
    CORSMiddleware is present in the app's middleware stack."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
