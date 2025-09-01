import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@pytest.mark.asyncio
async def test_read_root():
    async with TestClient(app) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "World"}
        print("Test passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_read_root())
