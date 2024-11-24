from app import app

def test_home_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"index.html" in response.data